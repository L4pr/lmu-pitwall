//! The player's best lap at the current track and car class, taken from
//! everything already stored in the results database.
//!
//! The dashboard needs a lap time to turn "minutes remaining" into "laps
//! remaining". Before the first timed lap of a session it has none of its own,
//! so it falls back to this.
//!
//! The lookup runs once per session, on a blocking thread, and the answer is
//! cached. The broadcaster only reads the cache, so the 1 Hz path never waits on
//! SQLite — not even while a first-run import holds the database lock for
//! seconds.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use rusqlite::{Connection, OptionalExtension};
use tracing::{debug, warn};

use super::importer::LAP_TIME_TOLERANCE;

/// Same plausibility window the fuel tracker applies to a live lap time.
const MIN_LAP: f64 = 10.0;
const MAX_LAP: f64 = 600.0;

/// Laps that must line up before a live session counts as proof that a result
/// file's track/class is the one shared memory is naming. One matching lap can
/// be a coincidence across hundreds of sessions; three in a row is not.
const MIN_MATCHING_LAPS: i64 = 3;

/// Fastest plausible lap the local player has driven at `track_name` in
/// `car_class`, in seconds. Both arguments are the strings shared memory
/// reports (`mTrackName`, `mVehicleClass`).
///
/// Three sources, all of our own laps only:
///
/// 1. Live captures (`live_laps`), keyed by exactly those strings.
/// 2. Result files whose `TrackCourse` and class spell the same — the common
///    case, where the game uses one name in both places.
/// 3. Result files proven to be the same track and class by a live session
///    whose lap times match one of their sessions lap for lap. This covers
///    history from before live capture existed, whatever the XML calls it.
pub fn lookup(conn: &Connection, track_name: &str, car_class: &str) -> Result<Option<f64>> {
    // CROSS JOIN pins the join order: start from our own driver rows (one per
    // session), walk laps(driver_id), then probe live_laps by track, class,
    // name, lap number and a lap-time range in one index hit. Left to itself the planner may scan
    // `drivers` once per live lap instead.
    let sql = "
        WITH proven AS (
            SELECT DISTINCT s.track_venue AS venue, s.track_course AS course, md.car_class AS class
            FROM (
                SELECT d.id AS driver_id
                FROM drivers d
                CROSS JOIN laps xl      ON xl.driver_id = d.id
                CROSS JOIN live_laps ll ON ll.player_name = d.name AND ll.lap_num = xl.lap_num
                WHERE d.is_self = 1
                  AND ll.track_name = ?1 AND ll.car_class = ?2
                  AND ll.lap_time BETWEEN xl.lap_time - ?3 AND xl.lap_time + ?3
                GROUP BY d.id, ll.session_key
                HAVING COUNT(*) >= ?4
            ) m
            JOIN drivers md ON md.id = m.driver_id
            JOIN sessions s ON s.id = md.session_id
        )
        SELECT MIN(t) FROM (
            SELECT lap_time AS t FROM live_laps
             WHERE track_name = ?1 AND car_class = ?2
            UNION ALL
            SELECT d.best_lap_time FROM sessions s
              JOIN drivers d ON d.session_id = s.id
             WHERE d.is_self = 1 AND s.track_course = ?1 AND d.car_class = ?2
            UNION ALL
            SELECT d.best_lap_time FROM proven p
              JOIN sessions s ON s.track_venue IS p.venue AND s.track_course IS p.course
              JOIN drivers d  ON d.session_id = s.id AND d.is_self = 1 AND d.car_class IS p.class
        )
        WHERE t > ?5 AND t < ?6";

    let best = conn
        .query_row(
            sql,
            rusqlite::params![
                track_name,
                car_class,
                LAP_TIME_TOLERANCE,
                MIN_MATCHING_LAPS,
                MIN_LAP,
                MAX_LAP
            ],
            |r| r.get::<_, Option<f64>>(0),
        )
        .optional()?
        .flatten();
    Ok(best)
}

/// First retry after a failed lookup; doubles per consecutive failure.
const RETRY_BASE: Duration = Duration::from_secs(5);
/// Ceiling for the retry interval, so a database that stays broken costs one
/// attempt (and one log line) every five minutes rather than every second.
const RETRY_MAX: Duration = Duration::from_secs(300);

type LookupFn = fn(&str, &str) -> Result<Option<f64>>;

/// The production lookup: the shared results database.
fn lookup_in_results_db(track: &str, class: &str) -> Result<Option<f64>> {
    let db = super::database::get_db()?;
    let conn = db.lock();
    lookup(&conn, track, class)
}

#[derive(Default)]
struct Slot {
    /// Bumped on every new lookup, so a slow answer for an old session cannot
    /// overwrite the cache after the session has moved on.
    generation: u64,
    value: Option<f64>,
    /// Consecutive failed lookups for the current session.
    failures: u32,
    /// When the last lookup failed; `None` while one is running or succeeded.
    failed_at: Option<Instant>,
}

/// Cached [`lookup`] result for the session currently on track.
pub struct HistoricalBestLap {
    key: String,
    last_session_et: f64,
    slot: Arc<Mutex<Slot>>,
    lookup: LookupFn,
    retry_base: Duration,
}

impl HistoricalBestLap {
    pub fn new() -> Self {
        Self::with_lookup(lookup_in_results_db, RETRY_BASE)
    }

    fn with_lookup(lookup: LookupFn, retry_base: Duration) -> Self {
        Self {
            key: String::new(),
            last_session_et: 0.0,
            slot: Arc::default(),
            lookup,
            retry_base,
        }
    }

    /// Call with the current session's identity as often as convenient — it is
    /// a string compare and an uncontended lock unless something changed. Must
    /// run inside a Tokio runtime.
    ///
    /// A lookup starts when:
    /// - the session, track or class changes;
    /// - the session clock (`session_et`, `mCurrentET`) runs backwards. A
    ///   restart, or the next session of the same type at the same track,
    ///   keeps every name the same, so this is what reveals it — and the new
    ///   lookup then includes the laps driven before it;
    /// - the previous lookup failed and its backoff has elapsed.
    pub fn observe(&mut self, session_key: &str, track_name: &str, car_class: &str, session_et: f64) {
        if track_name.is_empty() || car_class.is_empty() {
            return;
        }
        // The 1 s slack keeps sampling jitter from looking like a restart.
        let restarted = session_et + 1.0 < self.last_session_et;
        self.last_session_et = session_et;

        let key = format!("{session_key}|{track_name}|{car_class}");
        let new_session = restarted || key != self.key;
        if !new_session && !self.retry_due() {
            return;
        }
        self.key = key;
        self.start_lookup(track_name, car_class, new_session);
    }

    fn retry_due(&self) -> bool {
        let slot = self.slot.lock().expect("historical best mutex poisoned");
        slot.failed_at
            .is_some_and(|at| at.elapsed() >= backoff(self.retry_base, slot.failures))
    }

    fn start_lookup(&self, track_name: &str, car_class: &str, new_session: bool) {
        let generation = {
            let mut slot = self.slot.lock().expect("historical best mutex poisoned");
            slot.generation += 1;
            slot.value = None;
            slot.failed_at = None;
            if new_session {
                slot.failures = 0;
            }
            slot.generation
        };

        let slot = Arc::clone(&self.slot);
        let lookup = self.lookup;
        let retry_base = self.retry_base;
        let (track, class) = (track_name.to_owned(), car_class.to_owned());
        tokio::task::spawn_blocking(move || {
            let result = lookup(&track, &class);
            let mut slot = slot.lock().expect("historical best mutex poisoned");
            if slot.generation != generation {
                return; // the session moved on while this ran
            }
            match result {
                Ok(best) => {
                    debug!(track = %track, class = %class, best = ?best, "historical best lap");
                    slot.value = best;
                    slot.failures = 0;
                }
                Err(e) => {
                    slot.failures += 1;
                    slot.failed_at = Some(Instant::now());
                    warn!(
                        "Could not look up the historical best lap (retrying in {}s): {}",
                        backoff(retry_base, slot.failures).as_secs(),
                        e
                    );
                }
            }
        });
    }

    /// The cached best lap in seconds; `None` until the lookup has finished or
    /// when there is no history for this track and class.
    pub fn get(&self) -> Option<f64> {
        self.slot.lock().expect("historical best mutex poisoned").value
    }
}

/// `base`, doubled for every failure after the first, capped at [`RETRY_MAX`].
fn backoff(base: Duration, failures: u32) -> Duration {
    let doublings = failures.saturating_sub(1).min(16);
    base.saturating_mul(1 << doublings).min(RETRY_MAX)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::post_race::database::init_database;
    use std::path::Path;

    const ME: &str = "Mirco Gyr";

    fn db() -> Connection {
        init_database(Path::new(":memory:")).unwrap()
    }

    /// One result-file session with a single driver row; returns the driver id.
    fn xml_session(
        conn: &Connection,
        venue: &str,
        course: &str,
        name: &str,
        class: &str,
        is_self: bool,
        best: f64,
        laps: &[f64],
    ) -> i64 {
        conn.execute(
            "INSERT INTO sessions (file_hash, file_path, track_venue, track_course)
             VALUES (hex(randomblob(8)), 'x.xml', ?1, ?2)",
            rusqlite::params![venue, course],
        )
        .unwrap();
        let session_id = conn.last_insert_rowid();
        conn.execute(
            "INSERT INTO drivers (session_id, name, car_class, is_player, is_self, best_lap_time)
             VALUES (?1, ?2, ?3, ?4, ?4, ?5)",
            rusqlite::params![session_id, name, class, is_self, best],
        )
        .unwrap();
        let driver_id = conn.last_insert_rowid();
        for (i, t) in laps.iter().enumerate() {
            conn.execute(
                "INSERT INTO laps (driver_id, session_id, lap_num, lap_time) VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![driver_id, session_id, i as i64 + 1, t],
            )
            .unwrap();
        }
        driver_id
    }

    fn live_session(conn: &Connection, key: &str, track: &str, class: &str, laps: &[f64]) {
        for (i, t) in laps.iter().enumerate() {
            conn.execute(
                "INSERT INTO live_laps (session_key, recorded_at, track_name, player_name, car_class, lap_num, lap_time)
                 VALUES (?1, 0, ?2, ?3, ?4, ?5, ?6)",
                rusqlite::params![key, track, ME, class, i as i64 + 1, t],
            )
            .unwrap();
        }
    }

    #[test]
    fn no_history_means_no_answer() {
        assert_eq!(lookup(&db(), "Monza", "GT3").unwrap(), None);
    }

    #[test]
    fn live_laps_at_the_same_track_and_class_count() {
        let conn = db();
        live_session(&conn, "1/Monza/Race", "Monza", "GT3", &[110.0, 108.4, 109.1]);
        live_session(&conn, "2/Monza/Race", "Monza", "Hyper", &[95.0]);
        live_session(&conn, "1/Spa/Race", "Spa", "GT3", &[100.0]);
        assert_eq!(lookup(&conn, "Monza", "GT3").unwrap(), Some(108.4));
    }

    #[test]
    fn a_result_file_with_the_same_course_name_counts() {
        let conn = db();
        xml_session(&conn, "Autodromo Monza", "Monza", ME, "GT3", true, 107.9, &[]);
        assert_eq!(lookup(&conn, "Monza", "GT3").unwrap(), Some(107.9));
    }

    #[test]
    fn other_drivers_and_other_classes_never_count() {
        let conn = db();
        xml_session(&conn, "Autodromo Monza", "Monza", "Rival", "GT3", false, 100.0, &[]);
        xml_session(&conn, "Autodromo Monza", "Monza", ME, "Hyper", true, 95.0, &[]);
        assert_eq!(lookup(&conn, "Monza", "GT3").unwrap(), None);
    }

    #[test]
    fn implausible_times_are_ignored() {
        let conn = db();
        live_session(&conn, "1/Monza/Race", "Monza", "GT3", &[0.0, -1.0, 5.0, 108.0, 9999.0]);
        assert_eq!(lookup(&conn, "Monza", "GT3").unwrap(), Some(108.0));
    }

    /// Old results whose names match nothing live are found through a newer
    /// session that was captured live and matches its result file.
    #[test]
    fn a_matched_live_session_unlocks_older_results_under_other_names() {
        let conn = db();
        // Old result, before live capture: fastest ever.
        xml_session(&conn, "Monza Venue", "Monza GP", ME, "LMGT3", true, 106.5, &[]);
        // Recent session: in the result file and captured live, same laps.
        let laps = [109.2, 108.8, 108.9];
        xml_session(&conn, "Monza Venue", "Monza GP", ME, "LMGT3", true, 108.8, &laps);
        live_session(&conn, "3/Monza/Race", "Monza", "GT3", &laps);

        assert_eq!(lookup(&conn, "Monza", "GT3").unwrap(), Some(106.5));
    }

    #[test]
    fn a_coincidental_lap_or_two_does_not_link_tracks() {
        let conn = db();
        // Spa result sharing two lap times with the Monza capture by chance.
        xml_session(&conn, "Spa", "Spa GP", ME, "GT3", true, 90.0, &[109.2, 108.8, 140.0]);
        live_session(&conn, "3/Monza/Race", "Monza", "GT3", &[109.2, 108.8, 108.9]);
        assert_eq!(lookup(&conn, "Monza", "GT3").unwrap(), Some(108.8));
    }

    #[test]
    fn the_planner_walks_the_indexes() {
        let conn = db();
        let plan: Vec<String> = conn
            .prepare(
                "EXPLAIN QUERY PLAN
                 SELECT d.id FROM drivers d
                 CROSS JOIN laps xl      ON xl.driver_id = d.id
                 CROSS JOIN live_laps ll ON ll.player_name = d.name AND ll.lap_num = xl.lap_num
                 WHERE d.is_self = 1 AND ll.track_name = 'a' AND ll.car_class = 'b'
                   AND ll.lap_time BETWEEN xl.lap_time - 0.05 AND xl.lap_time + 0.05",
            )
            .unwrap()
            .query_map([], |r| r.get::<_, String>(3))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        let plan = plan.join("\n");
        assert!(plan.contains("SEARCH xl USING INDEX idx_laps_driver_id"), "{plan}");
        // One exact probe per lap, not a scan of every lap at the track.
        let ll = plan.lines().find(|l| l.contains(" ll ")).unwrap_or_default();
        assert!(ll.starts_with("SEARCH") && ll.contains("lap_time>?"), "{plan}");
    }

    // --- The cache around the lookup -------------------------------------

    use std::sync::atomic::{AtomicU32, Ordering};

    /// Waits for the background lookup(s) to settle.
    async fn until(mut done: impl FnMut() -> bool) {
        for _ in 0..400 {
            if done() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("background lookup never settled");
    }

    static ONCE_CALLS: AtomicU32 = AtomicU32::new(0);
    fn counting_once(_: &str, _: &str) -> Result<Option<f64>> {
        ONCE_CALLS.fetch_add(1, Ordering::SeqCst);
        Ok(Some(100.0))
    }

    #[tokio::test]
    async fn one_session_is_looked_up_once() {
        let mut cache = HistoricalBestLap::with_lookup(counting_once, RETRY_BASE);
        for et in [5.0, 6.0, 7.0, 300.0] {
            cache.observe("10/Monza", "Monza", "GT3", et);
        }
        until(|| cache.get().is_some()).await;
        assert_eq!(cache.get(), Some(100.0));
        assert_eq!(ONCE_CALLS.load(Ordering::SeqCst), 1);
    }

    static RESTART_CALLS: AtomicU32 = AtomicU32::new(0);
    fn counting_restart(_: &str, _: &str) -> Result<Option<f64>> {
        Ok(Some(100.0 + RESTART_CALLS.fetch_add(1, Ordering::SeqCst) as f64))
    }

    /// Same session number, track and class — only the clock shows that a
    /// new session began, and the laps from the previous one must count.
    #[tokio::test]
    async fn a_restart_with_identical_names_is_looked_up_again() {
        let mut cache = HistoricalBestLap::with_lookup(counting_restart, RETRY_BASE);
        cache.observe("10/Monza", "Monza", "GT3", 900.0);
        until(|| cache.get() == Some(100.0)).await;

        cache.observe("10/Monza", "Monza", "GT3", 3.0);
        until(|| cache.get() == Some(101.0)).await;
        assert_eq!(RESTART_CALLS.load(Ordering::SeqCst), 2);
    }

    static FLAKY_CALLS: AtomicU32 = AtomicU32::new(0);
    fn flaky(_: &str, _: &str) -> Result<Option<f64>> {
        match FLAKY_CALLS.fetch_add(1, Ordering::SeqCst) {
            0 => Err(anyhow::anyhow!("database is locked")),
            _ => Ok(Some(100.0)),
        }
    }

    #[tokio::test]
    async fn a_failed_lookup_is_retried_after_the_backoff() {
        let base = Duration::from_millis(40);
        let mut cache = HistoricalBestLap::with_lookup(flaky, base);
        cache.observe("10/Monza", "Monza", "GT3", 5.0);
        until(|| cache.slot.lock().unwrap().failed_at.is_some()).await;

        // Inside the backoff: no new attempt.
        cache.observe("10/Monza", "Monza", "GT3", 6.0);
        assert_eq!(FLAKY_CALLS.load(Ordering::SeqCst), 1);

        tokio::time::sleep(base).await;
        cache.observe("10/Monza", "Monza", "GT3", 7.0);
        until(|| cache.get().is_some()).await;
        assert_eq!(cache.get(), Some(100.0));
        assert_eq!(FLAKY_CALLS.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn the_backoff_doubles_up_to_five_minutes() {
        let s = |n| backoff(RETRY_BASE, n).as_secs();
        assert_eq!((s(1), s(2), s(3), s(4)), (5, 10, 20, 40));
        assert_eq!(s(7), 300);
        assert_eq!(s(u32::MAX), 300);
    }
}
