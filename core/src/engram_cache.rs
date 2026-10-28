//! DSX_ENGRAM_CACHE_GB=N: Engram table rows kept in process memory, one slot per row. The page cache keeps a whole
//! 4 KiB page for each ~264-byte row it has read (MADV_RANDOM, rows scattered by hash), so the same RAM holds ~15x more
//! rows here and fewer gathers wait on the drive. Rows are copied byte for byte: dequantized values do not change.
//! 8-way set-associative, CLOCK within a set, one mutex per shard of sets; memory is committed as slots fill.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

const WAYS: usize = 8;
const SHARDS: usize = 4096;

struct Shard {
    tags: Vec<u64>, // key + 1 per slot, 0 = empty
    refs: Vec<u8>,
    hand: Vec<u8>, // CLOCK hand per set
    data: Vec<u8>,
}

pub struct RowCache {
    rb: usize,
    sets_per_shard: usize,
    shards: Vec<Mutex<Shard>>,
    pub hits: AtomicU64,
    pub misses: AtomicU64,
}

static CACHE: OnceLock<Option<RowCache>> = OnceLock::new();

/// the process's row cache for rows of `rb` bytes, if DSX_ENGRAM_CACHE_GB is set (the first caller fixes `rb`)
pub fn get(rb: usize) -> Option<&'static RowCache> {
    CACHE
        .get_or_init(|| {
            let gb: f64 = std::env::var("DSX_ENGRAM_CACHE_GB").ok()?.parse().ok().filter(|&g: &f64| g > 0.0)?;
            let c = RowCache::new(rb, (gb * 1e9) as usize);
            eprintln!("engram row cache: {:.1} GB, {} rows of {rb} bytes", gb, c.capacity());
            Some(c)
        })
        .as_ref()
        .filter(|c| c.rb == rb)
}

/// (hits, misses) so far, when the cache is on
pub fn stats() -> Option<(u64, u64)> {
    CACHE.get().and_then(|c| c.as_ref()).map(|c| (c.hits.load(Relaxed), c.misses.load(Relaxed)))
}

impl RowCache {
    pub fn new(rb: usize, bytes: usize) -> RowCache {
        let slots = bytes / (rb + 9);
        let sets_per_shard = (slots / WAYS / SHARDS).max(1);
        let n = sets_per_shard * WAYS;
        // vec![0; n] comes from calloc: pages are committed only when a slot in them is first written
        let shards = (0..SHARDS).map(|_| Mutex::new(Shard { tags: vec![0; n], refs: vec![0; n], hand: vec![0; sets_per_shard], data: vec![0; n * rb] })).collect();
        RowCache { rb, sets_per_shard, shards, hits: AtomicU64::new(0), misses: AtomicU64::new(0) }
    }

    pub fn capacity(&self) -> usize {
        SHARDS * self.sets_per_shard * WAYS
    }

    fn place(&self, key: u64) -> (usize, usize) {
        let h = (key.wrapping_add(1)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let h = h ^ (h >> 29);
        ((h % SHARDS as u64) as usize, ((h / SHARDS as u64) % self.sets_per_shard as u64) as usize)
    }

    pub fn contains(&self, key: u64) -> bool {
        let (sh, set) = self.place(key);
        let s = self.shards[sh].lock().unwrap();
        s.tags[set * WAYS..(set + 1) * WAYS].contains(&(key + 1))
    }

    /// copy row `key` into `dst` if it is cached
    pub fn read(&self, key: u64, dst: &mut [u8]) -> bool {
        let (sh, set) = self.place(key);
        let mut s = self.shards[sh].lock().unwrap();
        for w in set * WAYS..(set + 1) * WAYS {
            if s.tags[w] == key + 1 {
                s.refs[w] = 1;
                dst.copy_from_slice(&s.data[w * self.rb..(w + 1) * self.rb]);
                drop(s);
                self.hits.fetch_add(1, Relaxed);
                return true;
            }
        }
        drop(s);
        self.misses.fetch_add(1, Relaxed);
        false
    }

    pub fn insert(&self, key: u64, src: &[u8]) {
        let (sh, set) = self.place(key);
        let mut s = self.shards[sh].lock().unwrap();
        let ways = set * WAYS..(set + 1) * WAYS;
        if s.tags[ways.clone()].contains(&(key + 1)) {
            return; // another thread filled it
        }
        let w = match s.tags[ways.clone()].iter().position(|&t| t == 0) {
            Some(i) => set * WAYS + i,
            None => loop {
                // CLOCK: the first slot from the hand whose reference bit is clear; bits are cleared on the way
                let i = set * WAYS + s.hand[set] as usize;
                s.hand[set] = ((s.hand[set] as usize + 1) % WAYS) as u8;
                if s.refs[i] == 0 {
                    break i;
                }
                s.refs[i] = 0;
            },
        };
        s.tags[w] = key + 1;
        s.refs[w] = 0;
        s.data[w * self.rb..(w + 1) * self.rb].copy_from_slice(src);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_round_trip_and_evict_within_capacity() {
        let rb = 264;
        let c = RowCache::new(rb, SHARDS * WAYS * 2 * (rb + 9)); // 2 sets per shard
        let row = |k: u64| (0..rb).map(|i| (k as usize * 31 + i) as u8).collect::<Vec<u8>>();
        let mut buf = vec![0u8; rb];
        assert!(!c.read(7, &mut buf));
        c.insert(7, &row(7));
        assert!(c.contains(7) && c.read(7, &mut buf) && buf == row(7));
        // far more keys than slots: every key still found is byte-exact, and the cache never exceeds its slots
        for k in 0..(c.capacity() as u64 * 4) {
            c.insert(k, &row(k));
        }
        let mut found = 0;
        for k in 0..(c.capacity() as u64 * 4) {
            if c.read(k, &mut buf) {
                assert_eq!(buf, row(k));
                found += 1;
            }
        }
        assert!(found > 0 && found <= c.capacity());
    }
}
