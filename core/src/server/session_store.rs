//! The disk tier behind the host-memory shelf of saved conversations (`sessions.rs`).
//!
//! One file per saved conversation: a 4 KiB header page, then the token key, the engine's metadata and the engine's
//! opaque state (compressed KV, index keys, window rings ...), each region padded to 4 KiB. The state moves with
//! O_DIRECT between the disk and pinned host memory, so saving or restoring a 600 MB conversation does not push the
//! page cache around on a host whose RAM is nearly all expert arena. A file is written under a temporary name,
//! fsynced and renamed: a crash leaves the old file or none, never half of one. CRC32s over every region and a
//! fingerprint of the weights and of the engine's state layout reject torn, damaged or foreign files.

use anyhow::{Context, Result, bail, ensure};
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::fs::{File, OpenOptions};
use crate::osio;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// O_DIRECT granularity: buffer addresses, file offsets and lengths are multiples of it.
pub const ALIGN: usize = 4096;
const MAGIC: &[u8; 8] = b"XSSESS\0\x01";
const VERSION: u32 = 1;
const EXT: &str = "xss";
const TMP: &str = ".tmp-";
/// unaligned writes are gathered into this many bytes before they go to the file
const BOUNCE: usize = 8 << 20;

/// What a file must match to be loaded: the weights and the engine's state layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fingerprint {
    pub weights: u64,
    pub layout: u64,
}

/// 64-bit FNV-1a, chained through `h` (start with `FNV0`).
pub const FNV0: u64 = 0xcbf2_9ce4_8422_2325;
pub fn fnv64(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// A file's identity for the fingerprint: its path, size and modification time.
pub fn file_identity(path: &Path) -> Result<u64> {
    let m = std::fs::metadata(path).with_context(|| format!("{}", path.display()))?;
    let t = m.modified()?.duration_since(UNIX_EPOCH).unwrap_or_default();
    let mut h = fnv64(FNV0, &osio::path_bytes(path));
    h = fnv64(h, &m.len().to_le_bytes());
    h = fnv64(h, &t.as_secs().to_le_bytes());
    Ok(fnv64(h, &t.subsec_nanos().to_le_bytes()))
}

pub fn pad(n: usize) -> usize {
    n.div_ceil(ALIGN) * ALIGN
}

pub fn unix_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

/// `t` (an Instant of this process) as unix milliseconds.
pub fn instant_to_unix_ms(t: Instant) -> u64 {
    unix_ms().saturating_sub(Instant::now().saturating_duration_since(t).as_millis() as u64)
}

/// Unix milliseconds as an Instant of this process: the future clamps to now; a past beyond the monotonic clock's
/// start (a file older than the uptime) lands as close to that start as it can, i.e. as old as possible.
pub fn unix_ms_to_instant(ms: u64) -> Instant {
    let now = Instant::now();
    let mut ago = Duration::from_millis(unix_ms().saturating_sub(ms));
    loop {
        if let Some(t) = now.checked_sub(ago) {
            return t;
        }
        ago /= 2;
    }
}

/// Zeroed, page-aligned heap memory (O_DIRECT needs aligned user buffers).
pub struct AlignedBuf {
    ptr: NonNull<u8>,
    len: usize,
}

// plain owned heap memory
unsafe impl Send for AlignedBuf {}

impl AlignedBuf {
    pub fn new(len: usize) -> Self {
        let len = pad(len.max(1));
        let layout = Layout::from_size_align(len, ALIGN).expect("aligned layout");
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) }).expect("aligned allocation");
        Self { ptr, len }
    }
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr.as_ptr(), Layout::from_size_align(self.len, ALIGN).unwrap()) }
    }
}

/// A loadable file, as its header and key describe it.
#[derive(Clone, Debug)]
pub struct FileInfo {
    pub path: PathBuf,
    pub client: String,
    pub key: Vec<u32>,
    pub pos: usize,
    pub rollback: Option<usize>,
    pub meta_len: usize,
    pub data_len: usize,
    /// bytes on disk (all regions, padded)
    pub file_bytes: u64,
    /// when the conversation was last used before it was written (unix ms)
    pub used_ms: u64,
}

#[derive(Clone, Debug)]
struct Header {
    fp: Fingerprint,
    client: String,
    pos: u64,
    rollback: Option<u64>,
    key_len: u64,
    meta_len: u64,
    data_len: u64,
    key_crc: u32,
    meta_crc: u32,
    data_crc: u32,
    used_ms: u64,
}

impl Header {
    fn encode(&self) -> AlignedBuf {
        let mut page = AlignedBuf::new(ALIGN);
        let b = page.as_mut_slice();
        b[0..8].copy_from_slice(MAGIC);
        b[8..12].copy_from_slice(&VERSION.to_le_bytes());
        b[16..24].copy_from_slice(&self.fp.weights.to_le_bytes());
        b[24..32].copy_from_slice(&self.fp.layout.to_le_bytes());
        b[32..40].copy_from_slice(&self.used_ms.to_le_bytes());
        b[40..48].copy_from_slice(&self.pos.to_le_bytes());
        b[48..56].copy_from_slice(&self.rollback.unwrap_or(u64::MAX).to_le_bytes());
        b[56..64].copy_from_slice(&self.key_len.to_le_bytes());
        b[64..72].copy_from_slice(&self.meta_len.to_le_bytes());
        b[72..80].copy_from_slice(&self.data_len.to_le_bytes());
        b[80..84].copy_from_slice(&self.key_crc.to_le_bytes());
        b[84..88].copy_from_slice(&self.meta_crc.to_le_bytes());
        b[88..92].copy_from_slice(&self.data_crc.to_le_bytes());
        let c = self.client.as_bytes();
        b[92..96].copy_from_slice(&(c.len() as u32).to_le_bytes());
        b[96..96 + c.len()].copy_from_slice(c);
        let crc = crc32fast::hash(b);
        b[12..16].copy_from_slice(&crc.to_le_bytes());
        page
    }

    fn decode(page: &[u8]) -> Result<Self> {
        ensure!(page.len() >= ALIGN && &page[0..8] == MAGIC, "not a session file");
        let u32_at = |o: usize| u32::from_le_bytes(page[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(page[o..o + 8].try_into().unwrap());
        ensure!(u32_at(8) == VERSION, "session file version {} (this engine reads {VERSION})", u32_at(8));
        let mut copy = page[..ALIGN].to_vec();
        copy[12..16].fill(0);
        ensure!(crc32fast::hash(&copy) == u32_at(12), "damaged header");
        let n = u32_at(92) as usize;
        ensure!(n <= 32, "damaged header (client name)");
        let rollback = u64_at(48);
        Ok(Self {
            fp: Fingerprint { weights: u64_at(16), layout: u64_at(24) },
            used_ms: u64_at(32),
            pos: u64_at(40),
            rollback: (rollback != u64::MAX).then_some(rollback),
            key_len: u64_at(56),
            meta_len: u64_at(64),
            data_len: u64_at(72),
            key_crc: u32_at(80),
            meta_crc: u32_at(84),
            data_crc: u32_at(88),
            client: String::from_utf8(page[96..96 + n].to_vec()).context("damaged header (client name)")?,
        })
    }

    fn key_bytes(&self) -> usize {
        self.key_len as usize * 4
    }
    fn data_off(&self) -> u64 {
        (ALIGN + pad(self.key_bytes()) + pad(self.meta_len as usize)) as u64
    }
    fn file_bytes(&self) -> u64 {
        self.data_off() + pad(self.data_len as usize) as u64
    }
}

static SEQ: AtomicU64 = AtomicU64::new(0);

/// A directory of session files for one set of weights and one state layout.
pub struct DiskStore {
    dir: PathBuf,
    fp: Fingerprint,
    reserve: u64,
    direct: bool,
}

impl DiskStore {
    /// Create the directory (owner-only: the files hold whole conversations) and check whether O_DIRECT works there.
    pub fn open(dir: &Path, fp: Fingerprint, reserve_bytes: u64) -> Result<Self> {
        std::fs::create_dir_all(dir).with_context(|| format!("session dir {}", dir.display()))?;
        osio::restrict_dir(dir)?;
        let probe = dir.join(format!("{TMP}probe-{}", std::process::id()));
        let mut probe_options = OpenOptions::new();
        probe_options.write(true).create(true).truncate(true);
        osio::set_direct(&mut probe_options);
        let direct = probe_options.open(&probe).is_ok();
        let _ = std::fs::remove_file(&probe);
        if !direct {
            tracing::warn!(dir = %dir.display(), "session dir: no O_DIRECT here, buffered I/O with page-cache drop");
        }
        Ok(Self { dir: dir.to_path_buf(), fp, reserve: reserve_bytes, direct })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Bytes the filesystem still gives this user.
    pub fn free_bytes(&self) -> Option<u64> {
        osio::free_bytes(&self.dir)
    }

    /// Every loadable file. Temporary leftovers and files that are damaged or belong to other weights or another
    /// state layout are deleted.
    pub fn scan(&self) -> Vec<FileInfo> {
        let mut out = vec![];
        let Ok(rd) = std::fs::read_dir(&self.dir) else { return out };
        for e in rd.flatten() {
            let path = e.path();
            let name = e.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(TMP) {
                let _ = std::fs::remove_file(&path);
                continue;
            }
            if path.extension().is_none_or(|x| x != EXT) {
                continue;
            }
            match self.open_reader(&path) {
                Ok(r) => out.push(r.info),
                Err(err) => {
                    tracing::warn!(file = %path.display(), error = %format!("{err:#}"), "session file dropped");
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
        out
    }

    fn open_file(&self, path: &Path, write: bool) -> Result<File> {
        osio::open_options(write, self.direct).open(path).with_context(|| format!("{}", path.display()))
    }

    /// Start a file: the key and metadata go out now, the state through `Writer::write`, the header at `finish`.
    #[allow(clippy::too_many_arguments)]
    pub fn create(&self, client: &str, key: &[u32], pos: usize, rollback: Option<usize>, used_ms: u64, meta: &[u8], data_len: usize) -> Result<Writer> {
        ensure!(client.len() <= 32, "client label longer than 32 bytes");
        let key_bytes: Vec<u8> = key.iter().flat_map(|t| t.to_le_bytes()).collect();
        let header = Header {
            fp: self.fp,
            client: client.into(),
            pos: pos as u64,
            rollback: rollback.map(|r| r as u64),
            key_len: key.len() as u64,
            meta_len: meta.len() as u64,
            data_len: data_len as u64,
            key_crc: crc32fast::hash(&key_bytes),
            meta_crc: crc32fast::hash(meta),
            data_crc: 0,
            used_ms,
        };
        let bytes = header.file_bytes();
        if let Some(free) = self.free_bytes() {
            if free < bytes + self.reserve {
                bail!("disk tier: {:.1} GB free, {:.2} GB file + {:.0} GB reserve does not fit", free as f64 / 1e9, bytes as f64 / 1e9, self.reserve as f64 / 1e9);
            }
        }
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let stem = format!("{client}-{}-{}-{n}", unix_ms(), std::process::id());
        let tmp = self.dir.join(format!("{TMP}{stem}.{EXT}"));
        let path = self.dir.join(format!("{stem}.{EXT}"));
        let f = self.open_file(&tmp, true)?;
        let w = Writer { f, direct: self.direct, tmp, path, dir: self.dir.clone(), off: header.data_off(), header, written: 0, crc: crc32fast::Hasher::new(), bounce: None, fill: 0, done: false };
        let kp = pad(key_bytes.len());
        let mut region = AlignedBuf::new(kp + pad(meta.len()));
        region.as_mut_slice()[..key_bytes.len()].copy_from_slice(&key_bytes);
        region.as_mut_slice()[kp..kp + meta.len()].copy_from_slice(meta);
        if kp + meta.len() > 0 {
            osio::write_all_at(&w.f, &region.as_slice()[..kp + pad(meta.len())], ALIGN as u64)?;
        }
        Ok(w)
    }

    /// Open a file for loading: header, fingerprint, key and metadata are checked; the state follows through `Reader`.
    pub fn open_reader(&self, path: &Path) -> Result<Reader> {
        let f = self.open_file(path, false)?;
        let mut page = AlignedBuf::new(ALIGN);
        osio::read_exact_at(&f, page.as_mut_slice(), 0).context("short header")?;
        let h = Header::decode(page.as_slice())?;
        ensure!(h.fp == self.fp, "written for other weights or another state layout");
        let len = f.metadata()?.len();
        ensure!(len >= h.file_bytes(), "truncated: {len} of {} bytes", h.file_bytes());
        let kb = h.key_bytes();
        let ml = h.meta_len as usize;
        let mut region = AlignedBuf::new(pad(kb) + pad(ml));
        if kb + ml > 0 {
            osio::read_exact_at(&f, &mut region.as_mut_slice()[..pad(kb) + pad(ml)], ALIGN as u64)?;
        }
        let r = region.as_slice();
        ensure!(crc32fast::hash(&r[..kb]) == h.key_crc, "damaged key");
        let meta = r[pad(kb)..pad(kb) + ml].to_vec();
        ensure!(crc32fast::hash(&meta) == h.meta_crc, "damaged metadata");
        let key = r[..kb].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        let info = FileInfo {
            path: path.to_path_buf(),
            client: h.client.clone(),
            key,
            pos: h.pos as usize,
            rollback: h.rollback.map(|r| r as usize),
            meta_len: ml,
            data_len: h.data_len as usize,
            file_bytes: h.file_bytes(),
            used_ms: h.used_ms,
        };
        Ok(Reader { f, direct: self.direct, off: h.data_off(), remaining: h.data_len as usize, data_crc: h.data_crc, crc: crc32fast::Hasher::new(), info, meta })
    }

    pub fn remove(&self, path: &Path) {
        if let Err(e) = std::fs::remove_file(path) {
            if e.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!(file = %path.display(), error = %e, "session file not removed");
            }
        }
    }
}

/// A session file being written. Dropped without `finish`, it leaves nothing behind.
pub struct Writer {
    f: File,
    direct: bool,
    tmp: PathBuf,
    path: PathBuf,
    dir: PathBuf,
    header: Header,
    /// file offset of the next state write
    off: u64,
    written: usize,
    crc: crc32fast::Hasher,
    bounce: Option<AlignedBuf>,
    fill: usize,
    done: bool,
}

impl Writer {
    /// Append state bytes. Whole aligned pages from an aligned buffer go straight to the file; the rest is gathered.
    pub fn write(&mut self, mut data: &[u8]) -> Result<()> {
        ensure!(self.written + data.len() <= self.header.data_len as usize, "more state than announced");
        self.crc.update(data);
        self.written += data.len();
        if self.fill == 0 && (data.as_ptr() as usize) % ALIGN == 0 {
            let whole = data.len() / ALIGN * ALIGN;
            if whole > 0 {
                osio::write_all_at(&self.f, &data[..whole], self.off)?;
                self.off += whole as u64;
                data = &data[whole..];
            }
        }
        while !data.is_empty() {
            let b = self.bounce.get_or_insert_with(|| AlignedBuf::new(BOUNCE));
            let n = data.len().min(BOUNCE - self.fill);
            b.as_mut_slice()[self.fill..self.fill + n].copy_from_slice(&data[..n]);
            self.fill += n;
            data = &data[n..];
            if self.fill == BOUNCE {
                self.flush_bounce()?;
            }
        }
        Ok(())
    }

    fn flush_bounce(&mut self) -> Result<()> {
        if self.fill == 0 {
            return Ok(());
        }
        let n = pad(self.fill);
        let b = self.bounce.as_mut().unwrap();
        b.as_mut_slice()[self.fill..n].fill(0);
        osio::write_all_at(&self.f, &b.as_slice()[..n], self.off)?;
        self.off += n as u64;
        self.fill = 0;
        Ok(())
    }

    /// Complete the file: last state bytes, header, fsync, rename, fsync of the directory. Returns path and size.
    pub fn finish(mut self) -> Result<(PathBuf, u64)> {
        ensure!(self.written == self.header.data_len as usize, "state: {} of {} bytes written", self.written, self.header.data_len);
        self.flush_bounce()?;
        self.header.data_crc = std::mem::take(&mut self.crc).finalize();
        osio::write_all_at(&self.f, self.header.encode().as_slice(), 0)?;
        self.f.sync_all()?;
        if !self.direct {
            osio::drop_page_cache(&self.f);
        }
        std::fs::rename(&self.tmp, &self.path)?;
        if let Ok(d) = File::open(&self.dir) {
            let _ = d.sync_all();
        }
        self.done = true;
        Ok((self.path.clone(), self.header.file_bytes()))
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        if !self.done {
            let _ = std::fs::remove_file(&self.tmp);
        }
    }
}

/// A session file being loaded: the state comes out in order through `read`, checked by `finish`.
pub struct Reader {
    f: File,
    direct: bool,
    off: u64,
    remaining: usize,
    data_crc: u32,
    crc: crc32fast::Hasher,
    pub info: FileInfo,
    pub meta: Vec<u8>,
}

impl Reader {
    /// The next state bytes into `buf` (page-aligned and a whole number of pages); returns how many it holds,
    /// 0 at the end.
    pub fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        if self.remaining == 0 {
            return Ok(0);
        }
        if self.direct {
            ensure!((buf.as_ptr() as usize) % ALIGN == 0 && buf.len() % ALIGN == 0 && !buf.is_empty(), "unaligned read buffer");
        }
        let n = buf.len().min(pad(self.remaining));
        osio::read_exact_at(&self.f, &mut buf[..n], self.off)?;
        self.off += n as u64;
        let got = n.min(self.remaining);
        self.crc.update(&buf[..got]);
        self.remaining -= got;
        Ok(got)
    }

    /// After the last `read`: every byte arrived and the state's CRC matches.
    pub fn finish(self) -> Result<()> {
        ensure!(self.remaining == 0, "state not read to the end");
        ensure!(self.crc.finalize() == self.data_crc, "damaged state (CRC mismatch)");
        if !self.direct {
            osio::drop_page_cache(&self.f);
        }
        Ok(())
    }
}
