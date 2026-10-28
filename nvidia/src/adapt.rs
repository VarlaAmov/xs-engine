//! Adaptive placement decisions, with no residency, worker flag or CUDA writes.

use crate::cache_select::Ranked;
use crate::gguf::GType;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

pub(crate) type Victims = HashMap<(GType, GType), Vec<Ranked>>;

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Swap {
    pub store: usize,
    pub victim: (usize, usize),
    pub new: (usize, usize),
}

/// The worker publishes this only after the last layer's picks and the entire score decay. The GPU's CPU/CPUD
/// flags precede decay, so even a logits download is not a score-completion fence.
#[derive(Default)]
pub(crate) struct ScoresReady(AtomicU32);

impl ScoresReady {
    pub fn publish(&self, seq: u32) {
        self.0.store(seq, Ordering::Release);
    }

    pub fn wait(&self, seq: u32) {
        while self.0.load(Ordering::Acquire) < seq {
            std::hint::spin_loop();
        }
    }
}

/// Candidates are hottest first; each same-type victim list has its coldest entry last. Consume virtual stage
/// capacity exactly as begin_swap would, leaving the real stores untouched until the token has finished.
pub(crate) fn choose(cand: &[Ranked], mut rooms: Vec<usize>, mut victims: Vec<Victims>, limit: usize,
    types: impl Fn(usize, usize) -> (GType, GType)) -> Vec<Swap> {
    let mut out = vec![];
    for &(v, l, e) in cand {
        if out.len() == limit {
            break;
        }
        let key = types(l, e);
        let best = (0..rooms.len())
            .filter(|&i| rooms[i] > 0)
            .filter_map(|i| victims[i].get(&key).and_then(|list| list.last()).map(|&(vs, _, _)| (vs, i)))
            .min_by(|a, b| a.0.total_cmp(&b.0));
        let Some((vs, i)) = best else { continue };
        if vs >= v * 0.5 {
            continue;
        }
        let (_, vl, ve) = victims[i].get_mut(&key).unwrap().pop().unwrap();
        rooms[i] -= 1;
        out.push(Swap { store: i, victim: (vl, ve), new: (l, e) });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const IQ: (GType, GType) = (GType::Iq2Xs, GType::Iq2Xs);
    const Q3: (GType, GType) = (GType::Q3K, GType::Q3K);

    #[test]
    fn planning_respects_types_stage_capacity_and_global_limit() {
        let cand = [(20.0, 0, 10), (18.0, 0, 11), (16.0, 0, 12), (14.0, 0, 13)];
        let victims = vec![
            HashMap::from([(IQ, vec![(3.0, 1, 1), (1.0, 1, 0)]), (Q3, vec![(0.0, 1, 2)])]),
            HashMap::from([(IQ, vec![(2.0, 2, 0)])]),
        ];
        let plan = choose(&cand, vec![1, 2], victims.clone(), 3, |_, e| if e == 11 { Q3 } else { IQ });
        assert_eq!(plan, vec![
            Swap { store: 0, victim: (1, 0), new: (0, 10) },
            Swap { store: 1, victim: (2, 0), new: (0, 12) },
        ]);
        // Planning did not evict anything, including the second victim in a store with only one free stage.
        assert_eq!(victims[0][&IQ].len(), 2);
        assert_eq!(choose(&cand, vec![2, 2], victims.clone(), 1, |_, _| IQ).len(), 1);
        assert!(choose(&cand, vec![2, 2], victims, 0, |_, _| IQ).is_empty());
    }

    #[test]
    fn hysteresis_and_store_ties_match_sequential_selection() {
        let cand = [(8.0, 0, 10), (4.0, 0, 11), (2.0, 0, 12)];
        let victims = vec![HashMap::from([(IQ, vec![(2.0, 1, 0)])]), HashMap::from([(IQ, vec![(2.0, 2, 0)])])];
        // Iterator::min_by chooses the first tied store; equality at the hysteresis threshold rejects a swap.
        assert_eq!(choose(&cand, vec![1, 1], victims, 3, |_, _| IQ),
            vec![Swap { store: 0, victim: (1, 0), new: (0, 10) }]);
    }

    #[test]
    fn a_commit_requires_a_new_plan() {
        let cand = [(10.0, 0, 10)];
        let before = vec![HashMap::from([(IQ, vec![(3.0, 1, 0)])])];
        let early = choose(&cand, vec![0], before.clone(), 1, |_, _| IQ);
        assert!(early.is_empty());
        // A completed pending swap both frees a stage and adds a resident: an early plan cannot be reused.
        let mut after = before;
        after[0].get_mut(&IQ).unwrap().push((1.0, 1, 1));
        assert_eq!(choose(&cand, vec![1], after, 1, |_, _| IQ),
            vec![Swap { store: 0, victim: (1, 1), new: (0, 10) }]);
    }

    #[test]
    fn score_epoch_publishes_all_layers_and_decay() {
        // Relaxed score stores become visible together after acquiring the worker's completed epoch.
        let ready = ScoresReady::default();
        let scores: Vec<AtomicU32> = (0..384).map(|_| AtomicU32::new(0)).collect();
        std::thread::scope(|s| {
            s.spawn(|| {
                for a in &scores {
                    a.store(17, Ordering::Relaxed);
                }
                ready.publish(40);
            });
            ready.wait(40);
            assert!(scores.iter().all(|a| a.load(Ordering::Relaxed) == 17));
        });
        ready.wait(39);
    }
}
