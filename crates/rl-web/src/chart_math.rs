//! Axis maths for charts: nice tick steps and tick positions.

/// A "nice" step (1, 2 or 5 × 10^k) that splits `range` into about `target` intervals.
pub fn nice_step(range: f64, target: usize) -> f64 {
    let range = if range > 0.0 { range } else { 1.0 };
    let raw = range / target.max(1) as f64;
    let magnitude = 10f64.powf(raw.log10().floor());
    let normalised = raw / magnitude;
    let nice = if normalised <= 1.0 {
        1.0
    } else if normalised <= 2.0 {
        2.0
    } else if normalised <= 5.0 {
        5.0
    } else {
        10.0
    };
    nice * magnitude
}

/// Tick values at multiples of a nice step inside `[min, max]`.
pub fn ticks(min: f64, max: f64, target: usize) -> Vec<f64> {
    let step = nice_step(max - min, target);
    let mut v = (min / step).ceil() * step;
    let mut out = Vec::new();
    while v <= max + step * 1e-9 {
        out.push(if v.abs() < step * 1e-9 { 0.0 } else { v });
        v += step;
    }
    out
}

/// Formats a tick or value with an explicit sign, dropping needless decimals.
pub fn signed(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 {
        let r = v.round() as i64;
        if r == 0 { "0".into() } else { format!("{r:+}") }
    } else {
        format!("{v:+.1}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_are_one_two_or_five_times_a_power_of_ten() {
        assert_eq!(nice_step(100.0, 5), 20.0);
        assert_eq!(nice_step(102.0, 12), 10.0);
        assert_eq!(nice_step(7.0, 5), 2.0);
        assert_eq!(nice_step(0.0, 5), 0.2);
    }

    #[test]
    fn ticks_stay_inside_the_range() {
        assert_eq!(ticks(-98.0, 2.0, 5), vec![-80.0, -60.0, -40.0, -20.0, 0.0]);
        assert_eq!(ticks(0.0, 4.0, 4), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn signed_formatting() {
        assert_eq!(signed(2.0), "+2");
        assert_eq!(signed(-57.25), "-57.2");
        assert_eq!(signed(0.0), "0");
    }
}
