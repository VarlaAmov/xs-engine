//! Persistent pinned worker pool: one job per call, all workers run it, spin-then-park between calls.
//! The caller does not take part; it spins until every worker is done.

use std::cell::UnsafeCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::*};
use std::thread::{self, JoinHandle, Thread};
use std::time::{Duration, Instant};

#[repr(align(128))]
#[derive(Default)]
struct Line<T>(T);

type Job = *const (dyn Fn(&Ctx) + Sync);

struct Shared {
    n: usize,
    epoch: Line<AtomicU64>,
    done: Line<AtomicUsize>,
    bar_count: Line<AtomicUsize>,
    bar_gen: Line<AtomicUsize>,
    sleeping: Vec<Line<AtomicBool>>,
    stop: AtomicBool,
    spin: Duration,
    job: UnsafeCell<Option<Job>>,
}
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

pub struct Ctx<'a> {
    pub id: usize,
    pub n: usize,
    sh: &'a Shared,
}

impl Ctx<'_> {
    /// Spin barrier across the pool's workers.
    pub fn barrier(&self) {
        let sh = self.sh;
        let g = sh.bar_gen.0.load(Acquire);
        if sh.bar_count.0.fetch_add(1, AcqRel) == sh.n - 1 {
            sh.bar_count.0.store(0, Relaxed);
            sh.bar_gen.0.store(g + 1, Release);
        } else {
            while sh.bar_gen.0.load(Acquire) == g {
                std::hint::spin_loop();
            }
        }
    }
}

pub struct Pool {
    sh: Arc<Shared>,
    threads: Vec<Thread>,
    handles: Vec<JoinHandle<()>>,
}

unsafe extern "C" {
    fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u64) -> i32;
}

/// Pin the calling thread to one logical CPU.
pub fn pin_to(cpu: usize) -> bool {
    pin_to_set(&[cpu])
}

/// restrict the calling thread to `cpus`
pub fn pin_to_set(cpus: &[usize]) -> bool {
    let mut set = [0u64; 16];
    for &cpu in cpus {
        if cpu >= 1024 { return false; }
        set[cpu / 64] |= 1 << (cpu % 64);
    }
    unsafe { sched_setaffinity(0, std::mem::size_of_val(&set), set.as_ptr()) == 0 }
}

// i7-12700K: CPUs 0-15 are the 8 P-cores (two threads each), 16-19 the E-cores. Every spinning thread gets a CPU
// of its own: two spinners on one CPU time-slice, and the loser sees its flag milliseconds late.
/// the thread that enqueues GPU work (alone on P-core 7; its sibling 15 stays nearly idle)
pub const MAIN_CPU: usize = 14;
/// the CPU expert worker's coordinator (spins on the route flag)
pub const WORKER_CPU: usize = 16;
/// the decode Engram prefetch and other helpers: off the pool's, the main thread's and the worker's CPUs
pub const AUX_CPUS: [usize; 4] = [15, 17, 18, 19];
/// every CPU (prefill helpers run while the expert pool sleeps)
pub const ALL_CPUS: [usize; 20] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19];

/// Logical CPUs 0..15: the 8 P-cores with their hyperthreads on the i7-12700K (and 13700).
pub const P_CORES: [usize; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// Where the spinning threads and the expert pool run. The defaults are the constants above (the i7-12700K); other
/// machines set them by environment, each a CPU list like `0-19,22`:
/// DSX_CPU_LIST (the CPU expert pool), DSX_CPU_MAIN (the thread enqueuing GPU work), DSX_CPU_WORKER (the expert
/// worker's coordinator), DSX_CPU_AUX (Engram and other helpers), DSX_CPU_ALL (prefill helpers; default: every CPU
/// this process may run on). Every spinning thread needs a CPU of its own: keep MAIN, WORKER and the pool disjoint.
pub struct Layout {
    /// None: DSX_CPU_THREADS (default 14) of P_CORES
    pub pool: Option<Vec<usize>>,
    pub main: usize,
    pub worker: usize,
    pub aux: Vec<usize>,
    pub all: Vec<usize>,
}

/// `0-3,8,10-11` -> [0, 1, 2, 3, 8, 10, 11]
pub fn parse_cpus(s: &str) -> Option<Vec<usize>> {
    let mut v = vec![];
    for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let (a, b) = match part.split_once('-') {
            Some((a, b)) => (a.trim().parse::<usize>().ok()?, b.trim().parse::<usize>().ok()?),
            None => { let c = part.parse().ok()?; (c, c) },
        };
        if a > b || b >= 1024 { return None; }
        for cpu in a..=b {
            if v.contains(&cpu) { return None; }
            v.push(cpu);
        }
    }
    (!v.is_empty() && v.iter().all(|&c| c < 1024)).then_some(v)
}

unsafe extern "C" {
    fn sched_getaffinity(pid: i32, cpusetsize: usize, mask: *mut u64) -> i32;
}

/// the CPUs this process may run on (at first call)
fn allowed() -> Vec<usize> {
    let mut set = [0u64; 16];
    if unsafe { sched_getaffinity(0, std::mem::size_of_val(&set), set.as_mut_ptr()) } != 0 {
        return ALL_CPUS.to_vec();
    }
    (0..1024).filter(|&c| set[c / 64] >> (c % 64) & 1 == 1).collect()
}

pub fn layout() -> &'static Layout {
    static L: std::sync::OnceLock<Layout> = std::sync::OnceLock::new();
    L.get_or_init(|| {
        let env = |k: &str| {
            std::env::var(k).ok().map(|v| parse_cpus(&v).unwrap_or_else(|| panic!("{k}={v:?}: expected a CPU list like 0-19,22")))
        };
        let one = |k: &str, d: usize| env(k).map_or(d, |v| v[0]);
        let l = Layout {
            pool: env("DSX_CPU_LIST"),
            main: one("DSX_CPU_MAIN", MAIN_CPU),
            worker: one("DSX_CPU_WORKER", WORKER_CPU),
            aux: env("DSX_CPU_AUX").unwrap_or_else(|| AUX_CPUS.to_vec()),
            all: env("DSX_CPU_ALL").unwrap_or_else(allowed),
        };
        if let Some(p) = &l.pool {
            for (name, c) in [("DSX_CPU_MAIN", l.main), ("DSX_CPU_WORKER", l.worker)] {
                if p.contains(&c) {
                    eprintln!("cpu layout: {name}={c} is also in DSX_CPU_LIST: two spinning threads share a CPU");
                }
            }
        }
        l
    })
}

impl Pool {
    /// One worker per entry of `cpus`, pinned there; workers spin `spin` after a job before parking.
    pub fn new(cpus: &[usize], spin: Duration) -> Pool {
        let n = cpus.len();
        assert!(n > 0);
        let sh = Arc::new(Shared {
            n,
            epoch: Line(AtomicU64::new(0)),
            done: Default::default(),
            bar_count: Default::default(),
            bar_gen: Default::default(),
            sleeping: (0..n).map(|_| Line::default()).collect(),
            stop: AtomicBool::new(false),
            spin,
            job: UnsafeCell::new(None),
        });
        let handles: Vec<JoinHandle<()>> = cpus
            .iter()
            .enumerate()
            .map(|(id, &cpu)| {
                let sh = sh.clone();
                thread::Builder::new()
                    .name(format!("cpu-exp-{id}"))
                    .spawn(move || worker(&sh, id, cpu))
                    .expect("spawn pool worker")
            })
            .collect();
        let threads = handles.iter().map(|h| h.thread().clone()).collect();
        Pool { sh, threads, handles }
    }

    pub fn threads(&self) -> usize {
        self.sh.n
    }

    /// Run `f` on every worker (ctx.id = 0..n) and wait for all of them.
    pub fn run<F: Fn(&Ctx) + Sync>(&self, f: &F) {
        let sh = &*self.sh;
        let job: *const (dyn Fn(&Ctx) + Sync + '_) = f;
        // lifetime erased: we do not return before every worker has finished with it
        unsafe { *sh.job.get() = Some(std::mem::transmute::<*const (dyn Fn(&Ctx) + Sync + '_), Job>(job)) };
        sh.done.0.store(0, Relaxed);
        sh.epoch.0.fetch_add(1, SeqCst);
        for (i, t) in self.threads.iter().enumerate() {
            if sh.sleeping[i].0.load(SeqCst) {
                t.unpark();
            }
        }
        while sh.done.0.load(Acquire) != sh.n {
            std::hint::spin_loop();
        }
    }
}

fn worker(sh: &Shared, id: usize, cpu: usize) {
    if !pin_to(cpu) {
        eprintln!("cpu pool: could not pin worker {id} to cpu {cpu}");
    }
    let mut seen = 0u64;
    loop {
        // wait for the next generation: spin, then park
        let t0 = Instant::now();
        let mut i = 0u32;
        loop {
            if sh.epoch.0.load(Acquire) != seen {
                break;
            }
            std::hint::spin_loop();
            i = i.wrapping_add(1);
            if i % 1024 == 0 && t0.elapsed() > sh.spin {
                sh.sleeping[id].0.store(true, SeqCst);
                if sh.epoch.0.load(SeqCst) == seen {
                    thread::park();
                }
                sh.sleeping[id].0.store(false, Relaxed);
            }
        }
        seen = sh.epoch.0.load(Acquire);
        if sh.stop.load(Acquire) {
            return;
        }
        let job = unsafe { (*sh.job.get()).unwrap() };
        let ctx = Ctx { id, n: sh.n, sh };
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe { (*job)(&ctx) })).is_err() {
            eprintln!("cpu pool worker {id} panicked");
            std::process::abort();
        }
        sh.done.0.fetch_add(1, Release);
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.sh.stop.store(true, Release);
        self.sh.epoch.0.fetch_add(1, SeqCst);
        for t in &self.threads {
            t.unpark();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_lists_bound_ranges_and_reject_duplicate_spinners() {
        assert_eq!(parse_cpus("0-2, 24-26"), Some(vec![0, 1, 2, 24, 25, 26]));
        for invalid in ["", "4-2", "0-1024", "0-18446744073709551615", "0-3,2", "1024", "-1", "abc"] {
            assert_eq!(parse_cpus(invalid), None, "{invalid}");
        }
        assert!(!pin_to_set(&[1024]));
    }

    #[test]
    fn runs_and_barriers() {
        let pool = Pool::new(&[0, 1, 2, 3], Duration::from_micros(200));
        let hits = AtomicUsize::new(0);
        for round in 0..200 {
            let before = hits.load(Relaxed);
            pool.run(&|c: &Ctx| {
                hits.fetch_add(1, Relaxed);
                c.barrier();
                // everyone incremented before anyone passes the barrier
                assert!(hits.load(Relaxed) >= before + c.n);
                c.barrier();
            });
            assert_eq!(hits.load(Relaxed), 4 * (round + 1));
            if round % 50 == 0 {
                thread::sleep(Duration::from_millis(2)); // let workers park
            }
        }
    }
}
