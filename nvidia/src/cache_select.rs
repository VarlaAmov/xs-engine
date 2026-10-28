//! Bounded cache ranking. The total order makes tied scores independent of HashMap iteration.
use std::cmp::Ordering;

pub(crate) type Ranked = (f32, usize, usize);

pub(crate) fn hottest(a: &Ranked, b: &Ranked) -> Ordering {
    b.0.total_cmp(&a.0).then_with(|| (a.1, a.2).cmp(&(b.1, b.2)))
}

pub(crate) fn coldest(a: &Ranked, b: &Ranked) -> Ordering {
    a.0.total_cmp(&b.0).then_with(|| (a.1, a.2).cmp(&(b.1, b.2)))
}

/// Keep exactly the same prefix as a full sort, doing no ordering work on unused entries.
pub(crate) fn keep(v: &mut Vec<Ranked>, n: usize, compare: fn(&Ranked, &Ranked) -> Ordering, full: bool) {
    if n == 0 {
        v.clear();
        return;
    }
    if !full && v.len() > n {
        v.select_nth_unstable_by(n, compare);
        v.truncate(n);
    }
    v.sort_unstable_by(compare);
    v.truncate(n);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_matches_full_sort_with_ties_and_all_limits() {
        let mut seed = 13u64;
        let mut input: Vec<Ranked> = (0..15360).map(|i| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (((seed >> 32) % 97) as f32 / 7.0, i / 384, i % 384)
        }).collect();
        // Resident iteration is deliberately different from layer/expert order.
        input.reverse();
        for cmp in [hottest as fn(&Ranked, &Ranked) -> Ordering, coldest] {
            for n in [0, 1, 8, 32, 3535, 15360, 20000] {
                let (mut full, mut bounded) = (input.clone(), input.clone());
                keep(&mut full, n, cmp, true);
                keep(&mut bounded, n, cmp, false);
                assert_eq!(full, bounded, "limit {n}");
            }
        }
        let mut empty = vec![];
        keep(&mut empty, 8, hottest, false);
        assert!(empty.is_empty());
    }

    #[test]
    #[ignore = "CPU microbenchmark; full-model timing is the acceptance criterion"]
    fn ranking_speed() {
        let input: Vec<Ranked> = (0..15360).map(|i| (((i * 7919 % 15401) as f32).sin().abs() * 100.0, i / 384, i % 384)).collect();
        for full in [true, false] {
            let start = std::time::Instant::now();
            for _ in 0..2000 {
                let mut v = input.clone();
                keep(&mut v, 32, hottest, full);
                std::hint::black_box(v);
            }
            println!("cache full={full}: {:.2} us/selection", start.elapsed().as_secs_f64() * 1e6 / 2000.0);
        }
    }
}
