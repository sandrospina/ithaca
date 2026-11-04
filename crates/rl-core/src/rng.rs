//! Deterministic random number generator.
//!
//! The algorithm is implemented here rather than taken from a crate, so a
//! dependency upgrade can never change the random stream. Replays are stored
//! as seed + action log and depend on this stream being stable forever.
//!
//! Algorithm: xoshiro256** seeded through SplitMix64. Integer-only, so it is
//! bit-identical on native and WebAssembly targets.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rng {
    s: [u64; 4],
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    /// Creates a generator from a 64-bit seed.
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let s = [splitmix64(&mut sm), splitmix64(&mut sm), splitmix64(&mut sm), splitmix64(&mut sm)];
        Self { s }
    }

    /// Derives an independent child generator, e.g. one per agent or per episode.
    pub fn fork(&mut self, stream: u64) -> Self {
        Self::new(self.next_u64() ^ stream.wrapping_mul(0xD1B5_4A32_D192_ED03))
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Uniform integer in `0..n`, without modulo bias. Panics if `n == 0`.
    pub fn below(&mut self, n: u32) -> u32 {
        assert!(n > 0, "Rng::below called with n = 0");
        // Lemire's multiply-shift with rejection.
        let threshold = n.wrapping_neg() % n;
        loop {
            let m = u64::from(self.next_u32()) * u64::from(n);
            if (m as u32) >= threshold {
                return (m >> 32) as u32;
            }
        }
    }

    /// Uniform float in `[0, 1)` built from 24 random bits; exact on every target.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / 16_777_216.0)
    }

    /// Returns true with probability `p`.
    pub fn chance(&mut self, p: f32) -> bool {
        self.next_f32() < p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::{prop_assert, proptest};

    #[test]
    fn same_seed_same_stream() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn stream_is_frozen() {
        // Guards against accidental algorithm changes: replays depend on these values.
        let mut r = Rng::new(0);
        let first: Vec<u64> = (0..3).map(|_| r.next_u64()).collect();
        let mut again = Rng::new(0);
        assert_eq!(first, (0..3).map(|_| again.next_u64()).collect::<Vec<_>>());
        assert_eq!(first, FROZEN_SEED0);
    }

    const FROZEN_SEED0: [u64; 3] = [0x99EC_5F36_CB75_F2B4, 0xBF6E_1F78_4956_452A, 0x1A5F_849D_4933_E6E0];

    #[test]
    fn below_is_roughly_uniform() {
        let mut r = Rng::new(7);
        let mut counts = [0u32; 4];
        for _ in 0..40_000 {
            counts[r.below(4) as usize] += 1;
        }
        for c in counts {
            assert!((9_000..11_000).contains(&c), "{counts:?}");
        }
    }

    proptest! {
        #[test]
        fn below_stays_in_range(seed: u64, n in 1u32..1000) {
            let mut r = Rng::new(seed);
            for _ in 0..100 {
                prop_assert!(r.below(n) < n);
            }
        }

        #[test]
        fn f32_in_unit_interval(seed: u64) {
            let mut r = Rng::new(seed);
            for _ in 0..100 {
                let x = r.next_f32();
                prop_assert!((0.0..1.0).contains(&x));
            }
        }
    }
}
