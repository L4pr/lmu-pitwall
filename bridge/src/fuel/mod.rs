use std::collections::VecDeque;

const MAX_SAMPLES: usize = 10;        // rolling window size for fuel and lap time samples
const PIT_FUEL_THRESHOLD: f64 = 5.0; // litres — fuel increase above this = pit stop
const PIT_FLASH_TICKS: u32 = 90;     // ~3 seconds at 30 Hz
const MAX_VALID_FUEL: f64 = 1000.0;  // no race car has > 1000 L tank
const MAX_VALID_LAP: i32 = 9_999;    // safety cap for mTotalLaps sentinel values
const MAX_VALID_CONSUMPTION: f64 = 100.0; // L/lap upper limit
/// In a race, the lap counter value once the first racing lap is complete.
/// LMU always uses a rolling start and counts the formation lap, so lap 1 is
/// the formation lap (already skipped as the stint's first lap) and lap 2 is
/// the start: traffic, a bunched field and no clean air — not race pace.
const FIRST_RACE_LAP: i32 = 2;

#[derive(Debug, Clone)]
pub struct FuelSnapshot {
    pub avg_consumption: f64,   // L/lap rolling median; 0.0 = no valid data yet
    pub sample_count: u32,      // number of fuel-consumption samples collected
    pub laps_remaining: f64,    // current_fuel / avg; Infinity if avg = 0
    pub stint_number: u32,      // 1-based, increments on each pit stop
    pub stint_laps: u32,        // laps completed in current stint (including outlap)
    pub stint_consumption: f64, // total fuel used since stint start
    pub recommended: f64,       // kept for protocol compatibility; always 0.0
    pub pit_detected: bool,     // true for ~3 s after a pit stop is detected
    pub avg_lap_time: f64,      // rolling median lap time (s); 0.0 = no valid data yet
    pub lap_time_count: u32,    // number of lap-time samples collected
}

impl Default for FuelSnapshot {
    fn default() -> Self {
        Self {
            avg_consumption: 0.0,
            sample_count: 0,
            laps_remaining: f64::INFINITY,
            stint_number: 1,
            stint_laps: 0,
            stint_consumption: 0.0,
            recommended: 0.0,
            pit_detected: false,
            avg_lap_time: 0.0,
            lap_time_count: 0,
        }
    }
}

pub struct FuelTracker {
    lap_fuel_samples: VecDeque<f64>,  // rolling per-lap consumption
    lap_time_samples: VecDeque<f64>,  // rolling per-lap times (seconds)
    fuel_at_lap_start: Option<f64>,   // fuel when current lap began
    prev_fuel: Option<f64>,           // fuel from previous valid tick (for pit detection)
    last_lap_number: i32,             // mTotalLaps on previous tick; -1 = uninitialised

    stint_number: u32,
    stint_start_lap: i32,
    stint_start_fuel: f64,
    stint_laps: u32,
    pit_flash_ticks: u32,             // counts down; pit_detected = true while > 0
    /// The car has been in the pit lane at some point during the current lap.
    /// Such a lap (in-lap, out-lap, drive-through) is not representative pace,
    /// so its time stays out of the average.
    pitted_this_lap: bool,
}

impl FuelTracker {
    pub fn new() -> Self {
        Self {
            lap_fuel_samples: VecDeque::with_capacity(MAX_SAMPLES),
            lap_time_samples: VecDeque::with_capacity(MAX_SAMPLES),
            fuel_at_lap_start: None,
            prev_fuel: None,
            last_lap_number: -1,
            stint_number: 1,
            stint_start_lap: 0,
            stint_start_fuel: 0.0,
            stint_laps: 0,
            pit_flash_ticks: 0,
            pitted_this_lap: false,
        }
    }

    /// Called every telemetry tick.
    ///
    /// * `current_fuel`          — mFuel for the player vehicle (litres)
    /// * `current_lap`           — mTotalLaps for the player vehicle (completed laps)
    /// * `in_pits`               — mInPits from scoring
    /// * `session_laps_remaining`— laps left; ≤ 0 = time-based / unknown
    /// * `last_lap_time`         — mLastLapTime from scoring (seconds); ≤ 0 = invalid
    /// * `is_race`               — race session; its first racing lap stays out of the lap-time average
    pub fn update(
        &mut self,
        current_fuel: f64,
        current_lap: i32,
        in_pits: bool,
        session_laps_remaining: i32,
        last_lap_time: f64,
        is_race: bool,
    ) -> FuelSnapshot {
        // Always count down pit-flash independently of input validity
        if self.pit_flash_ticks > 0 {
            self.pit_flash_ticks -= 1;
        }

        // --- Input validation: reject sentinel / garbage values ---
        // mTotalLaps is i16 → max 32767; mFuel should never be negative or huge
        let fuel_valid = current_fuel >= 0.0 && current_fuel <= MAX_VALID_FUEL;
        let lap_valid  = current_lap >= 0 && current_lap <= MAX_VALID_LAP;

        if !fuel_valid || !lap_valid {
            // Return snapshot based on last known good state without updating it
            let display_fuel = self.prev_fuel.unwrap_or(0.0);
            return self.build_snapshot(display_fuel);
        }

        if in_pits {
            self.pitted_this_lap = true;
        }

        // --- Pit stop detection (fuel spike while in pits) ---
        if let Some(prev) = self.prev_fuel {
            if in_pits && current_fuel > prev + PIT_FUEL_THRESHOLD {
                self.stint_number    += 1;
                self.stint_start_lap  = current_lap;
                self.stint_start_fuel = current_fuel;
                self.stint_laps       = 0;
                self.lap_fuel_samples.clear();
                self.lap_time_samples.clear();
                self.fuel_at_lap_start = Some(current_fuel);
                self.pit_flash_ticks   = PIT_FLASH_TICKS;
            }
        }
        self.prev_fuel = Some(current_fuel);

        // --- Session restart detection ---
        // A restart of the same session keeps the same session key, so the outer
        // session-change guard in main.rs does not fire.  The reliable signal is
        // mTotalLaps dropping back below the last recorded lap number.  When that
        // happens, discard all stale samples and re-initialise from scratch.
        if self.last_lap_number >= 0 && current_lap < self.last_lap_number {
            self.lap_fuel_samples.clear();
            self.lap_time_samples.clear();
            self.fuel_at_lap_start = None;
            self.prev_fuel         = Some(current_fuel);
            self.last_lap_number   = -1;
            self.stint_number      = 1;
            self.stint_start_lap   = current_lap;
            self.stint_start_fuel  = current_fuel;
            self.stint_laps        = 0;
            self.pit_flash_ticks   = 0;
            self.pitted_this_lap   = in_pits;
        }

        // --- First valid update: initialise tracker state ---
        if self.last_lap_number < 0 {
            self.last_lap_number   = current_lap;
            self.fuel_at_lap_start = Some(current_fuel);
            self.stint_start_fuel  = current_fuel;
            self.stint_start_lap   = current_lap;
        }

        // --- Lap transition ---
        if current_lap > self.last_lap_number {
            // Skip the first lap of each stint (outlap / formation lap).
            // self.stint_laps holds the count BEFORE this transition, so
            // 0 means the outlap just finished → skip its data.
            // For stint_laps >= 1, the lap that just finished is a real racing lap → record.
            if self.stint_laps >= 1 {
                // Fuel consumption sample
                if let Some(lap_start_fuel) = self.fuel_at_lap_start {
                    let consumed = lap_start_fuel - current_fuel;
                    if consumed > 0.0 && consumed < MAX_VALID_CONSUMPTION {
                        if self.lap_fuel_samples.len() >= MAX_SAMPLES {
                            self.lap_fuel_samples.pop_front();
                        }
                        self.lap_fuel_samples.push_back(consumed);
                    }
                }
                // Lap time sample — not for a lap that went through the pit
                // lane, nor for the opening lap of a race.
                let race_start_lap = is_race && current_lap <= FIRST_RACE_LAP;
                if !self.pitted_this_lap && !race_start_lap
                    && last_lap_time > 10.0 && last_lap_time < 600.0
                {
                    if self.lap_time_samples.len() >= MAX_SAMPLES {
                        self.lap_time_samples.pop_front();
                    }
                    self.lap_time_samples.push_back(last_lap_time);
                }
            }

            self.last_lap_number   = current_lap;
            self.fuel_at_lap_start = Some(current_fuel);
            self.stint_laps = (current_lap - self.stint_start_lap).max(0) as u32;
            // Crossing the line in the pit lane makes the new lap an out-lap.
            self.pitted_this_lap = in_pits;
        }

        // Suppress unused warning — session_laps_remaining kept for potential future use
        let _ = session_laps_remaining;

        self.build_snapshot(current_fuel)
    }

    fn build_snapshot(&self, current_fuel: f64) -> FuelSnapshot {
        let avg = median(&self.lap_fuel_samples);
        let laps_remaining = if avg > 0.0 {
            current_fuel / avg
        } else {
            f64::INFINITY
        };
        let stint_consumption = (self.stint_start_fuel - current_fuel).max(0.0);
        let avg_lap_time = median(&self.lap_time_samples);

        FuelSnapshot {
            avg_consumption:  avg,
            sample_count:     self.lap_fuel_samples.len() as u32,
            laps_remaining,
            stint_number:     self.stint_number,
            stint_laps:       self.stint_laps,
            stint_consumption,
            recommended:      0.0,
            pit_detected:     self.pit_flash_ticks > 0,
            avg_lap_time,
            lap_time_count:   self.lap_time_samples.len() as u32,
        }
    }
}

fn median(samples: &VecDeque<f64>) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = samples.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drives one lap: a tick mid-lap (optionally in the pit lane), then the
    /// line crossing that completes it with `lap_time`.
    fn lap(t: &mut FuelTracker, lap: i32, fuel: f64, mid_in_pits: bool, cross_in_pits: bool, lap_time: f64) -> FuelSnapshot {
        t.update(fuel + 1.0, lap - 1, mid_in_pits, -1, -1.0, false);
        t.update(fuel, lap, cross_in_pits, -1, lap_time, false)
    }

    fn race_lap(t: &mut FuelTracker, lap: i32, fuel: f64, lap_time: f64) -> FuelSnapshot {
        t.update(fuel + 1.0, lap - 1, false, -1, -1.0, true);
        t.update(fuel, lap, false, -1, lap_time, true)
    }

    fn race_start() -> FuelTracker {
        let mut t = FuelTracker::new();
        t.update(100.0, 0, false, -1, -1.0, false);
        lap(&mut t, 1, 97.0, false, false, 130.0); // first lap of the stint: always skipped
        t
    }

    #[test]
    fn racing_laps_make_the_average() {
        let mut t = race_start();
        lap(&mut t, 2, 94.0, false, false, 100.0);
        let snap = lap(&mut t, 3, 91.0, false, false, 102.0);
        assert_eq!(snap.lap_time_count, 2);
        assert_eq!(snap.avg_lap_time, 101.0);
    }

    /// A stop without refuelling (drive-through, tyres only) is never seen as a
    /// new stint, so before this both of its laps counted.
    #[test]
    fn in_lap_and_out_lap_of_a_stop_without_fuel_are_left_out() {
        let mut t = race_start();
        lap(&mut t, 2, 94.0, false, false, 100.0);
        lap(&mut t, 3, 91.0, true, true, 125.0);   // in-lap: crosses the line in the pit lane
        lap(&mut t, 4, 88.0, false, false, 118.0); // out-lap: began in the pit lane
        let snap = lap(&mut t, 5, 85.0, false, false, 102.0);
        assert_eq!(snap.lap_time_count, 2);
        assert_eq!(snap.avg_lap_time, 101.0);
    }

    /// Pit box past the line: the lap ends on track but contained the pit lane.
    #[test]
    fn a_lap_that_only_passed_through_the_pit_lane_is_left_out() {
        let mut t = race_start();
        lap(&mut t, 2, 94.0, false, false, 100.0);
        let snap = lap(&mut t, 3, 91.0, true, false, 140.0);
        assert_eq!(snap.lap_time_count, 1);
        assert_eq!(snap.avg_lap_time, 100.0);
    }

    #[test]
    fn the_opening_lap_of_a_race_is_left_out() {
        let mut t = FuelTracker::new();
        t.update(100.0, 0, false, -1, -1.0, true);
        race_lap(&mut t, 1, 97.0, 150.0); // formation lap
        race_lap(&mut t, 2, 94.0, 112.0); // the start
        race_lap(&mut t, 3, 91.0, 100.0);
        let snap = race_lap(&mut t, 4, 88.0, 102.0);
        assert_eq!(snap.lap_time_count, 2);
        assert_eq!(snap.avg_lap_time, 101.0);
    }

    /// Practice and qualifying have no standing field to fight through.
    #[test]
    fn outside_a_race_lap_two_counts() {
        let mut t = race_start();
        let snap = lap(&mut t, 2, 94.0, false, false, 100.0);
        assert_eq!(snap.lap_time_count, 1);
    }
}
