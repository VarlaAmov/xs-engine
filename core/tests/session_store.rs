//! Session files on a real filesystem (the test target dir: O_DIRECT where it is supported, buffered elsewhere).
use dsx_core::server::session_store::{ALIGN, AlignedBuf, DiskStore, Fingerprint, pad};
use std::path::PathBuf;

const FP: Fingerprint = Fingerprint { weights: 0x1111, layout: 0x2222 };

fn dir(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&d);
    d
}

/// pseudo-random bytes (xorshift), so a misplaced page shows
fn bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect()
}

/// Read the whole state back through aligned buffers of `chunk` bytes.
fn read_all(store: &DiskStore, path: &std::path::Path, chunk: usize) -> anyhow::Result<(Vec<u8>, Vec<u8>, Vec<u32>)> {
    let mut r = store.open_reader(path)?;
    let mut buf = AlignedBuf::new(chunk);
    let mut out = vec![];
    loop {
        let n = r.read(&mut buf.as_mut_slice()[..chunk])?;
        if n == 0 {
            break;
        }
        out.extend_from_slice(&buf.as_slice()[..n]);
    }
    let (meta, key) = (r.meta.clone(), r.info.key.clone());
    r.finish()?;
    Ok((out, meta, key))
}

#[test]
fn roundtrip_with_uneven_writes_and_reads() {
    let d = dir("store-roundtrip");
    let store = DiskStore::open(&d, FP, 0).unwrap();
    let key: Vec<u32> = (0..12_345).collect();
    let meta = bytes(777, 3);
    let data = bytes(5 * ALIGN + 1234, 7);
    let mut w = store.create("owner", &key, key.len(), Some(12_000), 42, &meta, data.len()).unwrap();
    // an aligned run of whole pages (straight to the file), then odd pieces (gathered)
    let mut aligned = AlignedBuf::new(2 * ALIGN);
    aligned.as_mut_slice().copy_from_slice(&data[..2 * ALIGN]);
    w.write(aligned.as_slice()).unwrap();
    w.write(&data[2 * ALIGN..2 * ALIGN + 100]).unwrap();
    w.write(&data[2 * ALIGN + 100..]).unwrap();
    let (path, size) = w.finish().unwrap();
    assert_eq!(size, std::fs::metadata(&path).unwrap().len());
    assert_eq!(size as usize, ALIGN + pad(key.len() * 4) + pad(meta.len()) + pad(data.len()));
    for chunk in [ALIGN, 3 * ALIGN, 64 * ALIGN] {
        let (out, m, k) = read_all(&store, &path, chunk).unwrap();
        assert_eq!(out, data, "chunk {chunk}");
        assert_eq!(m, meta);
        assert_eq!(k, key);
    }
    let found = store.scan();
    assert_eq!(found.len(), 1);
    let f = &found[0];
    assert_eq!((f.client.as_str(), f.pos, f.rollback, f.used_ms, f.data_len), ("owner", 12_345, Some(12_000), 42, data.len()));
    // owner-only: the files hold whole conversations
    use std::os::unix::fs::PermissionsExt;
    assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
    assert_eq!(std::fs::metadata(&d).unwrap().permissions().mode() & 0o777, 0o700);
}

#[test]
fn damage_and_foreign_files_are_caught() {
    let d = dir("store-damage");
    let store = DiskStore::open(&d, FP, 0).unwrap();
    let key: Vec<u32> = (0..100).collect();
    let data = bytes(3 * ALIGN, 11);
    let write = |client: &str| {
        let mut w = store.create(client, &key, key.len(), None, 1, b"meta", data.len()).unwrap();
        w.write(&data).unwrap();
        w.finish().unwrap().0
    };
    // one flipped bit in the state: the CRC catches it at the end of the read
    let p = write("owner");
    let mut raw = std::fs::read(&p).unwrap();
    let off = raw.len() - 2 * ALIGN + 5;
    raw[off] ^= 0x10;
    std::fs::write(&p, &raw).unwrap();
    assert!(read_all(&store, &p, ALIGN).unwrap_err().to_string().contains("CRC"));
    // a damaged header, a truncated file, another model's file, a temporary leftover: scan removes them all
    let h = write("friend");
    let mut raw = std::fs::read(&h).unwrap();
    raw[50] ^= 1;
    std::fs::write(&h, &raw).unwrap();
    let t = write("owner");
    let raw = std::fs::read(&t).unwrap();
    std::fs::write(&t, &raw[..raw.len() - ALIGN]).unwrap();
    let other = DiskStore::open(&d, Fingerprint { weights: 9, layout: 0x2222 }, 0).unwrap();
    let mut w = other.create("owner", &key, key.len(), None, 1, b"", data.len()).unwrap();
    w.write(&data).unwrap();
    let foreign = w.finish().unwrap().0;
    std::fs::write(d.join(".tmp-owner-1-2-3.xss"), b"half").unwrap();
    let good = write("owner");
    let found = store.scan();
    let names: Vec<_> = found.iter().map(|f| f.path.clone()).collect();
    assert!(names.contains(&good), "the intact file stays");
    assert!(names.contains(&p), "the bit flip in the state is only seen when the state is read");
    for gone in [&h, &t, &foreign] {
        assert!(!gone.exists(), "{} should be removed", gone.display());
    }
    assert!(!d.join(".tmp-owner-1-2-3.xss").exists());
}

#[test]
fn an_unfinished_writer_leaves_nothing_and_the_reserve_is_kept() {
    let d = dir("store-unfinished");
    let store = DiskStore::open(&d, FP, 0).unwrap();
    {
        let mut w = store.create("owner", &[1, 2, 3], 3, None, 1, b"", 10 * ALIGN).unwrap();
        w.write(&bytes(ALIGN, 1)).unwrap();
        // dropped without finish (an error or a crash in the middle)
    }
    assert_eq!(std::fs::read_dir(&d).unwrap().count(), 0);
    // more state than announced is refused
    let mut w = store.create("owner", &[1], 1, None, 1, b"", 10).unwrap();
    assert!(w.write(&[0u8; 11]).is_err());
    drop(w);
    // a reserve larger than the disk: nothing is created
    let full = DiskStore::open(&d, FP, u64::MAX / 2).unwrap();
    assert!(full.create("owner", &[1], 1, None, 1, b"", 10).is_err());
    assert_eq!(std::fs::read_dir(&d).unwrap().count(), 0);
}
