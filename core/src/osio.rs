//! The few file operations the engine needs that std spells differently on Unix and Windows: positional
//! reads and writes, unbuffered (O_DIRECT / FILE_FLAG_NO_BUFFERING) opens, owner-only permissions, free space,
//! a path's raw bytes and the page-cache drop after a buffered write. Unix keeps the original calls exactly.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;

/// The path's bytes for hashing: the OS string as stored (UTF-16 code units, little-endian, on Windows).
pub fn path_bytes(path: &Path) -> Vec<u8> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        path.as_os_str().as_bytes().to_vec()
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        path.as_os_str().encode_wide().flat_map(u16::to_le_bytes).collect()
    }
}

/// Owner-only access to a directory. On Windows a directory under the user's profile already inherits an
/// owner-only ACL, so this leaves the inherited ACL as it is.
pub fn restrict_dir(dir: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
    }
    #[cfg(windows)]
    {
        let _ = dir;
        Ok(())
    }
}

/// Options for a new owner-only file (`write`) or a read, optionally bypassing the page cache.
pub fn open_options(write: bool, direct: bool) -> OpenOptions {
    let mut options = OpenOptions::new();
    if write {
        options.write(true).create_new(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    } else {
        options.read(true);
    }
    if direct {
        set_direct(&mut options);
    }
    options
}

/// Bypass the page cache: O_DIRECT on Linux, FILE_FLAG_NO_BUFFERING on Windows. Both need offsets, lengths
/// and buffer addresses aligned to the sector size, which the 4 KiB `ALIGN` of the callers covers.
pub fn set_direct(options: &mut OpenOptions) {
    #[cfg(target_os = "linux")]
    std::os::unix::fs::OpenOptionsExt::custom_flags(options, libc::O_DIRECT);
    #[cfg(windows)]
    {
        const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
        std::os::windows::fs::OpenOptionsExt::custom_flags(options, FILE_FLAG_NO_BUFFERING);
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    let _ = options;
}

/// Write all of `buf` at `offset` without moving a shared cursor.
pub fn write_all_at(file: &File, mut buf: &[u8], mut offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    {
        let _ = (&mut buf, &mut offset);
        std::os::unix::fs::FileExt::write_all_at(file, buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        while !buf.is_empty() {
            let n = file.seek_write(buf, offset)?;
            if n == 0 {
                return Err(io::Error::new(io::ErrorKind::WriteZero, "seek_write wrote nothing"));
            }
            buf = &buf[n..];
            offset += n as u64;
        }
        Ok(())
    }
}

/// One positional read into `buf` at `offset`; returns how many bytes arrived (0 at end of file).
pub fn read_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    #[cfg(unix)]
    {
        std::os::unix::fs::FileExt::read_at(file, buf, offset)
    }
    #[cfg(windows)]
    {
        std::os::windows::fs::FileExt::seek_read(file, buf, offset)
    }
}

/// Fill all of `buf` from `offset`, or fail with UnexpectedEof.
pub fn read_exact_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    {
        let _ = (&mut buf, &mut offset);
        std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        while !buf.is_empty() {
            let n = file.seek_read(buf, offset)?;
            if n == 0 {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short read"));
            }
            buf = &mut buf[n..];
            offset += n as u64;
        }
        Ok(())
    }
}

/// Bytes the filesystem holding `dir` still gives this user.
pub fn free_bytes(dir: &Path) -> Option<u64> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let c = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
        let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
        (unsafe { libc::statvfs(c.as_ptr(), &mut st) } == 0).then(|| st.f_bavail as u64 * st.f_frsize as u64)
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetDiskFreeSpaceExW(dir: *const u16, free_to_caller: *mut u64, total: *mut u64, free: *mut u64) -> i32;
        }
        let wide: Vec<u16> = dir.as_os_str().encode_wide().chain(Some(0)).collect();
        let mut free_to_caller = 0u64;
        // SAFETY: `wide` is NUL-terminated and outlives the call; the out-pointers are valid u64s or null.
        let ok = unsafe { GetDiskFreeSpaceExW(wide.as_ptr(), &mut free_to_caller, std::ptr::null_mut(), std::ptr::null_mut()) };
        (ok != 0).then_some(free_to_caller)
    }
}

/// Drop a buffered file's pages from the cache after writing or reading it once. Windows keeps no such hint for
/// an open handle; the standby list ages out on its own.
pub fn drop_page_cache(file: &File) {
    #[cfg(target_os = "linux")]
    unsafe {
        libc::posix_fadvise(std::os::fd::AsRawFd::as_raw_fd(file), 0, 0, libc::POSIX_FADV_DONTNEED);
    }
    #[cfg(not(target_os = "linux"))]
    let _ = file;
}
