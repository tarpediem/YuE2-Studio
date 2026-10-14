//! The CUDA libraries the music engine is linked against.
//!
//! The engine loads one of two CUDA backends at run time, `cuda13\ggml-cuda.dll`
//! or `cuda12\ggml-cuda.dll`, whichever the card and its driver run (see
//! `hardware::CudaBuild`). Each imports its own cuBLAS, `cublas64_13.dll` or
//! `cublas64_12.dll`, which imports `cublasLt64_1x.dll`. The loader resolves
//! them from the executable's folder, so they go beside `yue-server.exe`, and
//! the two majors never clash by name. A missing cuBLAS fails the backend
//! load, and the engine stops on it.
//!
//! The two libraries are 512 MB unpacked, which is four times the rest of the
//! installer, so they are not shipped - they are fetched from NVIDIA's own
//! redistributable archive the first time the engine is asked to start, the
//! same way the models are. The build machine has them on PATH through the CUDA
//! Toolkit, which is exactly why this was invisible until someone without the
//! Toolkit ran the release.

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
#[cfg(test)]
use anyhow::Context;

use crate::downloads::{Asset, AssetKind, Downloader};
use crate::hardware::CudaBuild;

/// The libraries of each build, by file name, exactly as its ggml-cuda.dll
/// imports them. The CUDA major is part of the name, so a rebuild on another
/// major is a change here, and the dependency check below catches it.
pub const CUDA13_LIBRARIES: [&str; 2] = ["cublas64_13.dll", "cublasLt64_13.dll"];
pub const CUDA12_LIBRARIES: [&str; 2] = ["cublas64_12.dll", "cublasLt64_12.dll"];

/// The Visual C++ runtime the engine and ggml are compiled against, by the
/// names in their import tables.
#[cfg(windows)]
pub const VC_RUNTIME_LIBRARIES: [&str; 4] =
    ["vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "vcomp140.dll"];


#[cfg(test)]
pub const ASSETS: &[Asset] = &[CUBLAS13, CUBLAS12];

const CUBLAS13: Asset = Asset {
    id: "engine-cuda-cublas",
    label: "NVIDIA cuBLAS 13.5",
    kind: AssetKind::Runtime,
    // NVIDIA's redistributable archive, the same one every CUDA application
    // ships from. The size and the digest are published in
    // redist/redistrib_13.3.0.json; the size below was confirmed against a
    // live HEAD request, which also confirmed range support.
    url: "https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-13.5.1.27-archive.zip",
    relative_path: "cuda-cublas.zip",
    bytes: 391_055_517,
    // No sub-directory: the libraries go straight in beside yue-server.exe,
    // which is where the Windows loader looks first and what NVIDIA's own
    // deployment guide recommends.
    unzip_into: None,
    marker: "cublasLt64_13.dll",
    pick: &CUDA13_LIBRARIES,
    vram_gb: None,
    note: "The linear algebra the engine's CUDA backend is linked against. Without it the engine cannot start at all.",
};

const CUBLAS12: Asset = Asset {
    id: "engine-cuda12-cublas",
    label: "NVIDIA cuBLAS 12.9",
    kind: AssetKind::Runtime,
    // The cuBLAS of the CUDA 12.9 toolkit the CUDA 12 backend is built with.
    // Size from redist/redistrib_12.9.1.json, confirmed by a HEAD request.
    url: "https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-12.9.1.4-archive.zip",
    relative_path: "cuda12-cublas.zip",
    bytes: 549_755_186,
    unzip_into: None,
    marker: "cublasLt64_12.dll",
    pick: &CUDA12_LIBRARIES,
    vram_gb: None,
    note: "The linear algebra of the engine's CUDA 12 backend, for cards CUDA 13 dropped and older drivers.",
};

/// The cuBLAS a CUDA build loads.
pub fn cublas_asset(build: CudaBuild) -> &'static Asset {
    match build {
        CudaBuild::Cuda13 => &CUBLAS13,
        CudaBuild::Cuda12 => &CUBLAS12,
    }
}

/// The engine's downloadable runtime.
pub struct EngineRuntime {
    downloader: Downloader,
}

impl EngineRuntime {
    /// Takes the directory `yue-server.exe` lives in: the libraries belong
    /// beside the binary that imports them, not in a folder of their own.
    pub fn new(bundle_root: &Path) -> Self {
        Self { downloader: Downloader::new(bundle_root.to_path_buf()) }
    }

    pub fn downloader(&self) -> &Downloader {
        &self.downloader
    }

    /// Where the libraries live once installed - beside the engine.
    #[cfg(test)]
    pub fn library_dir(&self) -> PathBuf {
        self.downloader.root().to_path_buf()
    }

    /// Whether the engine will find every library it needs. cuBLAS counts only
    /// when the engine will compute on CUDA, and only the one of that build:
    /// the bundled engine loads its backends at run time, so on an AMD or
    /// Intel card, or with Vulkan chosen, no CUDA backend is ever loaded.
    ///
    /// The Visual C++ runtime counts always: a machine that already has cuBLAS
    /// but no redistributable has nothing to download and still cannot start
    /// the engine.
    pub fn is_ready(&self, cuda: Option<CudaBuild>) -> bool {
        self.missing(cuda).is_empty() && self.vc_runtime_missing().is_empty() && !system_cublas_missing(cuda)
    }

    /// The Visual C++ runtime ships inside the engine bundle, app-local as
    /// Microsoft permits, so the studio never installs anything into the
    /// system. A machine that has it on its search path is fine either way.
    #[cfg(windows)]
    pub fn vc_runtime_missing(&self) -> Vec<&'static str> {
        VC_RUNTIME_LIBRARIES
            .iter()
            .copied()
            .filter(|library| !self.downloader.root().join(library).is_file() && !is_on_the_search_path(library))
            .collect()
    }

    /// Off Windows the engine is built against the platform's own runtime
    /// (libc++ and the system libraries), which is always present: nothing to
    /// check, nothing to fetch.
    #[cfg(not(windows))]
    pub fn vc_runtime_missing(&self) -> Vec<&'static str> {
        Vec::new()
    }

    /// What is still missing, so a caller can report the size before starting.
    ///
    /// A machine that already has the libraries on its search path - a CUDA
    /// Toolkit installation - downloads nothing: the engine inherits that path
    /// and finds them there.
    pub fn missing(&self, cuda: Option<CudaBuild>) -> Vec<&'static Asset> {
        // The cuBLAS archives are Windows builds: on Linux the system CUDA toolkit provides it, and
        // `system_cublas_missing` says when it does not.
        if cfg!(target_os = "linux") {
            return Vec::new();
        }
        let Some(build) = cuda else { return Vec::new() };
        let asset = cublas_asset(build);
        if self.downloader.is_installed(asset) || asset.pick.iter().all(|library| is_on_the_search_path(library)) {
            return Vec::new();
        }
        vec![asset]
    }

    /// Fetches whatever is missing and waits for it.
    pub async fn install_missing(&self, cuda: Option<CudaBuild>) -> Result<()> {
        if system_cublas_missing(cuda) {
            bail!("{SYSTEM_CUDA_REQUIRED}");
        }
        let absent = self.vc_runtime_missing();
        if !absent.is_empty() {
            bail!("the engine bundle is incomplete: {} missing beside yue-server.exe; reinstall the studio", absent.join(", "));
        }
        self.downloader.install_all("engine", &self.missing(cuda)).await
    }
}

/// Why a Linux machine without cuBLAS cannot compute on CUDA. Nothing is downloaded there.
pub const SYSTEM_CUDA_REQUIRED: &str =
    "the system CUDA toolkit is required: install the `cuda` package so that libcublas.so and libcublasLt.so are on the library path";

/// The cuBLAS files each CUDA build links against on Linux, by soname. Only the fallback: the sonames the
/// shipped `libggml-cuda.so` really asks for are read from its dynamic section.
#[cfg(target_os = "linux")]
fn linux_libraries(build: CudaBuild) -> [&'static str; 2] {
    match build {
        CudaBuild::Cuda13 => ["libcublas.so.13", "libcublasLt.so.13"],
        CudaBuild::Cuda12 => ["libcublas.so.12", "libcublasLt.so.12"],
    }
}

/// The directories the dynamic loader always searches. The toolkit's own folder (`/opt/cuda/lib64`) is
/// not one of them: it counts only through `LD_LIBRARY_PATH` or the loader cache.
#[cfg(target_os = "linux")]
const SYSTEM_LIBRARY_DIRECTORIES: [&str; 3] = ["/usr/lib", "/usr/lib64", "/usr/lib/x86_64-linux-gnu"];

/// Whether a library sits in one of the directories.
#[cfg(target_os = "linux")]
fn in_directories(library: &str, directories: impl IntoIterator<Item = std::path::PathBuf>) -> bool {
    directories.into_iter().any(|directory| directory.join(library).is_file())
}

/// The loader cache as `ldconfig -p` prints it, read at most every few seconds: the setup status is
/// polled every second, and installing the toolkit while the studio runs must still be noticed.
#[cfg(target_os = "linux")]
fn loader_cache() -> String {
    use std::sync::Mutex;
    use std::time::{Duration, Instant};
    static CACHE: Mutex<Option<(Instant, String)>> = Mutex::new(None);
    let mut cache = CACHE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some((read_at, text)) = cache.as_ref() {
        if read_at.elapsed() < Duration::from_secs(20) {
            return text.clone();
        }
    }
    let text = ["ldconfig", "/sbin/ldconfig", "/usr/sbin/ldconfig"]
        .iter()
        .find_map(|program| std::process::Command::new(program).arg("-p").output().ok().filter(|output| output.status.success()))
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
        .unwrap_or_default();
    *cache = Some((Instant::now(), text.clone()));
    text
}

/// Whether the loader finds the library: `LD_LIBRARY_PATH`, its default folders, then the loader cache
/// (which also holds the toolkit's folder where the package registered it in `ld.so.conf.d`).
#[cfg(target_os = "linux")]
fn is_findable_on_linux(library: &str) -> bool {
    let from_environment = std::env::var_os("LD_LIBRARY_PATH").map(|paths| std::env::split_paths(&paths).filter(|path| !path.as_os_str().is_empty()).collect::<Vec<_>>()).unwrap_or_default();
    if in_directories(library, from_environment.into_iter().chain(SYSTEM_LIBRARY_DIRECTORIES.iter().map(std::path::PathBuf::from))) {
        return true;
    }
    loader_cache().lines().any(|line| line.trim_start().starts_with(library))
}

/// The shared libraries an ELF file lists as needed (`DT_NEEDED`), read from its dynamic section.
/// None when the file is not a 64-bit little-endian ELF.
#[cfg(target_os = "linux")]
pub fn needed_libraries(path: &Path) -> Option<Vec<String>> {
    elf_needed(&std::fs::read(path).ok()?)
}

#[cfg(target_os = "linux")]
fn elf_needed(data: &[u8]) -> Option<Vec<String>> {
    let number = |offset: usize, width: usize| -> Option<u64> {
        let bytes = data.get(offset..offset.checked_add(width)?)?;
        Some(bytes.iter().rev().fold(0u64, |value, byte| (value << 8) | u64::from(*byte)))
    };
    // ELF64, little endian
    if data.get(..4)? != b"\x7fELF" || *data.get(4)? != 2 || *data.get(5)? != 1 {
        return None;
    }
    let (phoff, phentsize, phnum) = (number(0x20, 8)? as usize, number(0x36, 2)? as usize, number(0x38, 2)? as usize);
    let mut dynamic = None;
    let mut loads = Vec::new();
    for index in 0..phnum {
        let header = phoff.checked_add(index.checked_mul(phentsize)?)?;
        let (kind, offset, address, size) = (number(header, 4)?, number(header + 8, 8)? as usize, number(header + 16, 8)?, number(header + 32, 8)? as usize);
        match kind {
            1 => loads.push((address, number(header + 40, 8)?, offset)),
            2 => dynamic = Some((offset, size)),
            _ => {}
        }
    }
    let (dynamic_offset, dynamic_size) = dynamic?;
    let to_offset = |address: u64| loads.iter().find(|(start, length, _)| address >= *start && address < start + length).map(|(start, _, offset)| offset + (address - start) as usize);
    let (mut needed, mut strings) = (Vec::new(), None);
    for entry in 0..dynamic_size / 16 {
        let at = dynamic_offset + entry * 16;
        let (tag, value) = (number(at, 8)?, number(at + 8, 8)?);
        match tag {
            0 => break,
            1 => needed.push(value as usize),
            5 => strings = to_offset(value),
            _ => {}
        }
    }
    let strings = strings?;
    needed
        .into_iter()
        .map(|name| {
            let tail = data.get(strings.checked_add(name)?..)?;
            let end = tail.iter().position(|byte| *byte == 0)?;
            Some(String::from_utf8_lossy(&tail[..end]).into_owned())
        })
        .collect()
}

/// The CUDA toolkit libraries the shipped CUDA backend needs, by what its own dynamic section says;
/// None when the backend is not beside the studio (a development run), and the build's assumed
/// sonames apply.
#[cfg(target_os = "linux")]
fn backend_toolkit_libraries() -> Option<Vec<String>> {
    static NEEDED: std::sync::OnceLock<Option<Vec<String>>> = std::sync::OnceLock::new();
    NEEDED
        .get_or_init(|| {
            let backend = bundled_resource("yue2-cpp", "libggml-cuda.so").or_else(|| std::env::var_os("YUE_ENGINE_ROOT").map(|root| PathBuf::from(root).join("libggml-cuda.so")).filter(|path| path.is_file()))?;
            let needed = needed_libraries(&backend)?;
            let toolkit: Vec<String> = needed.into_iter().filter(|name| ["libcublas", "libcudart", "libnccl"].iter().any(|prefix| name.starts_with(prefix))).collect();
            (!toolkit.is_empty()).then_some(toolkit)
        })
        .clone()
}

/// The libraries that are missing for the CUDA backend of this build, by soname.
#[cfg(target_os = "linux")]
fn missing_linux_libraries(build: CudaBuild) -> Vec<String> {
    let wanted = backend_toolkit_libraries().unwrap_or_else(|| linux_libraries(build).iter().map(|name| (*name).to_string()).collect());
    wanted.into_iter().filter(|library| !is_findable_on_linux(library)).collect()
}

/// Linux only: whether the system's CUDA toolkit libraries for this CUDA build cannot be found. Always
/// false off Linux, and when no CUDA build runs. The CUDA device is unavailable then: Auto moves on to
/// Vulkan or the processor, and only an explicit choice of CUDA reports `SYSTEM_CUDA_REQUIRED`.
pub fn system_cublas_missing(cuda: Option<CudaBuild>) -> bool {
    #[cfg(target_os = "linux")]
    if let Some(build) = cuda {
        return !missing_linux_libraries(build).is_empty();
    }
    let _ = cuda;
    false
}

/// A tool the Linux build carries beside the studio, `<executable folder>/resources/<folder>/<file>`.
pub fn bundled_resource(folder: &str, file: &str) -> Option<PathBuf> {
    let tool = std::env::current_exe().ok()?.parent()?.join("resources").join(folder).join(file);
    tool.is_file().then_some(tool)
}

/// The search path a native tool needs to find its libraries, as the variable to set on the child and
/// its value. Windows: the engine's folder (where its CUDA runtime lives) goes first on `PATH`. Linux:
/// the tool carries its own ggml, so its own folder goes first on `LD_LIBRARY_PATH` and the engine's
/// copy is never the one it loads; the system loader finds the CUDA toolkit by itself. None when there
/// is nothing to add.
pub fn library_search_path(tool: &Path, engine_libraries: Option<&Path>) -> Option<(&'static str, std::ffi::OsString)> {
    let (variable, first) = if cfg!(target_os = "linux") { ("LD_LIBRARY_PATH", tool.parent()?) } else { ("PATH", engine_libraries?) };
    join_search_path(first, std::env::var_os(variable)).map(|value| (variable, value))
}

/// `first`, then the entries of an existing search path. Empty entries are dropped: on Linux an empty
/// one means the current directory.
fn join_search_path(first: &Path, existing: Option<std::ffi::OsString>) -> Option<std::ffi::OsString> {
    let mut paths = vec![first.to_path_buf()];
    paths.extend(existing.iter().flat_map(std::env::split_paths).filter(|path| !path.as_os_str().is_empty()));
    std::env::join_paths(paths).ok()
}

/// Whether the loader would already find this library without help.
///
/// The engine inherits this process's PATH, so the directories searched there
/// are the directories searched here. A machine with the CUDA Toolkit
/// installed has cuBLAS on PATH already, and asking it to download half a
/// gigabyte of the same thing would be rude.
fn is_on_the_search_path(library: &str) -> bool {
    let Some(path) = std::env::var_os("PATH") else { return false };
    std::env::split_paths(&path).any(|directory| directory.join(library).is_file())
}

/// The DLLs a Windows binary names in its import table.
///
/// NVIDIA's deployment guide says to read this with `dumpbin /IMPORTS` and
/// redistribute exactly what it lists, because the binary compatibility
/// version is part of the file name - `cublas64_13.dll`, not `cublas.dll`.
/// Reading it here means the release can check itself instead of trusting
/// that whoever built it remembered.
#[cfg(test)]
pub fn imported_libraries(binary: &Path) -> Result<Vec<String>> {
    let data = std::fs::read(binary)?;
    let at = |offset: usize| -> Result<u32> {
        let bytes = data.get(offset..offset + 4).context("truncated PE header")?;
        Ok(u32::from_le_bytes(bytes.try_into().expect("four bytes")))
    };
    let short = |offset: usize| -> Result<u16> {
        let bytes = data.get(offset..offset + 2).context("truncated PE header")?;
        Ok(u16::from_le_bytes(bytes.try_into().expect("two bytes")))
    };

    let pe = at(0x3c)? as usize;
    if data.get(pe..pe + 4) != Some(b"PE\0\0") {
        bail!("{} is not a PE binary", binary.display());
    }
    let sections = short(pe + 6)? as usize;
    let optional_size = short(pe + 20)? as usize;
    let optional = pe + 24;
    // 0x20b is PE32+, whose data directories start 16 bytes further in.
    let directories = optional + if short(optional)? == 0x20b { 112 } else { 96 };
    let import_rva = at(directories + 8)?;
    if import_rva == 0 {
        return Ok(Vec::new());
    }

    let headers: Vec<(u32, u32, u32)> = (0..sections)
        .map(|index| {
            let base = optional + optional_size + index * 40;
            Ok((at(base + 12)?, at(base + 16)?, at(base + 20)?))
        })
        .collect::<Result<_>>()?;
    let offset_of = |rva: u32| -> Option<usize> {
        headers
            .iter()
            .find(|(virtual_address, size, _)| rva >= *virtual_address && rva < virtual_address + size)
            .map(|(virtual_address, _, raw)| (raw + (rva - virtual_address)) as usize)
    };

    let mut names = Vec::new();
    let mut entry = offset_of(import_rva).context("import table points outside the file")?;
    loop {
        let descriptor = data.get(entry..entry + 20).context("truncated import table")?;
        if descriptor.iter().all(|byte| *byte == 0) {
            break;
        }
        let name_rva = u32::from_le_bytes(descriptor[12..16].try_into().expect("four bytes"));
        let start = offset_of(name_rva).context("an import name points outside the file")?;
        let end = data[start..].iter().position(|byte| *byte == 0).context("unterminated import name")? + start;
        names.push(String::from_utf8_lossy(&data[start..end]).into_owned());
        entry += 20;
    }
    Ok(names)
}

/// Libraries every Windows machine has, or that arrive with the display
/// driver. Everything else has to be shipped or downloaded.
#[cfg(test)]
fn is_provided_by_the_system(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name.starts_with("api-ms-win-")
        || name.starts_with("ext-ms-win-")
        // The display driver's own library: NVIDIA's guide is explicit that
        // this one is never redistributed - the user installs a driver.
        || name == "nvcuda.dll"
        || [
            "kernel32.dll",
            "kernelbase.dll",
            "user32.dll",
            "advapi32.dll",
            "shell32.dll",
            "ole32.dll",
            "oleaut32.dll",
            "ws2_32.dll",
            "crypt32.dll",
            "bcrypt.dll",
            "ntdll.dll",
            "rpcrt4.dll",
            "setupapi.dll",
            "cfgmgr32.dll",
            "gdi32.dll",
            "version.dll",
            "dbghelp.dll",
            "powrprof.dll",
            "psapi.dll",
            "userenv.dll",
            "winmm.dll",
            "msvcrt.dll",
        ]
        .contains(&name.as_str())
}

/// What a binary needs that is neither beside it nor supplied by Windows.
///
/// Follows the chain: `yue-server.exe` imports `ggml.dll`, which imports
/// `ggml-cuda.dll`, which is where cuBLAS actually comes in. Checking only the
/// executable's own imports would have found nothing wrong with the release
/// that could not start.
#[cfg(test)]
pub fn unresolved_dependencies(directory: &Path, entry_point: &str) -> Result<Vec<String>> {
    let mut seen: Vec<String> = Vec::new();
    let mut queue = vec![entry_point.to_string()];
    let mut missing = Vec::new();
    while let Some(name) = queue.pop() {
        let lowered = name.to_ascii_lowercase();
        if seen.contains(&lowered) {
            continue;
        }
        seen.push(lowered);
        let path = directory.join(&name);
        if !path.is_file() {
            missing.push(name);
            continue;
        }
        for import in imported_libraries(&path)? {
            if !is_provided_by_the_system(&import) {
                queue.push(import);
            }
        }
    }
    Ok(missing)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn a_library_is_found_in_the_directory_that_holds_it() {
        let root = std::env::temp_dir().join(format!("yue-cublas-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("libcublas.so.13"), b"").unwrap();
        assert!(in_directories("libcublas.so.13", [root.clone()]));
        assert!(!in_directories("libcublasLt.so.13", [root.clone()]));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn empty_search_path_entries_are_dropped() {
        let joined = join_search_path(Path::new("/opt/tool"), Some(":/usr/lib::".into())).unwrap();
        assert_eq!(std::env::split_paths(&joined).collect::<Vec<_>>(), vec![PathBuf::from("/opt/tool"), PathBuf::from("/usr/lib")]);
        let alone = join_search_path(Path::new("/opt/tool"), Some("".into())).unwrap();
        assert_eq!(std::env::split_paths(&alone).collect::<Vec<_>>(), vec![PathBuf::from("/opt/tool")]);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn the_needed_libraries_are_read_from_an_elf_file() {
        let own = needed_libraries(&std::env::current_exe().unwrap()).expect("the test binary is an ELF file");
        assert!(own.iter().any(|name| name.starts_with("libc.so")), "{own:?}");
        let root = std::env::temp_dir().join(format!("yue-elf-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("not-elf"), b"MZ plainly not an ELF file").unwrap();
        assert_eq!(needed_libraries(&root.join("not-elf")), None);
        assert_eq!(needed_libraries(&root.join("missing")), None);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_tool_looks_in_its_own_folder_on_linux_and_the_engines_elsewhere() {
        let (variable, value) = library_search_path(Path::new("/opt/tool/music-train"), Some(Path::new("/opt/engine"))).unwrap();
        let first = std::env::split_paths(&value).next();
        if cfg!(target_os = "linux") {
            assert_eq!((variable, first), ("LD_LIBRARY_PATH", Some(PathBuf::from("/opt/tool"))));
        } else {
            assert_eq!((variable, first), ("PATH", Some(PathBuf::from("/opt/engine"))));
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn nothing_is_ever_downloaded_for_cublas_on_linux() {
        let runtime = EngineRuntime::new(&std::env::temp_dir());
        assert!(runtime.missing(Some(CudaBuild::Cuda13)).is_empty());
        assert!(runtime.missing(Some(CudaBuild::Cuda12)).is_empty());
    }

    #[cfg(not(target_os = "linux"))]
    #[test]
    fn the_system_cublas_is_never_asked_for_off_linux() {
        assert!(!system_cublas_missing(Some(CudaBuild::Cuda13)));
    }

    /// The file names are the contract with the engine binary: they are what
    /// the import table asks the loader for, and what `pick` takes out of the
    /// archive. A typo here is a download that finishes and changes nothing.
    #[test]
    fn every_imported_library_is_picked_out_of_the_archive() {
        for (build, libraries, major) in
            [(CudaBuild::Cuda13, CUDA13_LIBRARIES, "13"), (CudaBuild::Cuda12, CUDA12_LIBRARIES, "12")]
        {
            let asset = cublas_asset(build);
            for library in libraries {
                assert!(library.contains(major), "{library} does not name CUDA {major}");
                assert!(asset.pick.contains(&library), "{library} is imported by {build:?} but never taken out of its archive");
            }
            assert!(asset.url.contains(&format!("-{major}.")), "{} is not a CUDA {major} cuBLAS", asset.url);
        }
    }

    /// A runtime asset that unpacks nowhere would download and vanish.
    #[test]
    fn the_libraries_land_in_one_directory_the_engine_can_be_pointed_at() {
        for asset in ASSETS {
            // Straight into the bundle: a sub-directory would put them
            // somewhere the loader does not look.
            assert_eq!(asset.unzip_into, None);
            assert_eq!(asset.kind, AssetKind::Runtime);
            assert!(asset.bytes > 0, "{} has no size, so nothing can report progress", asset.id);
        }
    }

    /// The release that shipped without cuBLAS passed every test there was,
    /// because no test ever looked at what the engine binary asks the loader
    /// for. This one does: whatever the staged bundle imports is either beside
    /// it, supplied by Windows, or downloaded on first start - and nothing
    /// else is allowed.
    #[test]
    #[cfg(windows)]
    fn the_staged_engine_bundle_can_actually_be_loaded() {
        let bundle = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../desktop/src-tauri/resources/yue2-cpp")
            .canonicalize();
        let Ok(bundle) = bundle else {
            // No engine has been built into the bundle on this machine, which
            // is the normal state of a checkout that only touches the service.
            return;
        };
        if !bundle.join("yue-server.exe").is_file() {
            return;
        }
        // The CUDA backends load at run time from their folders and resolve
        // their imports from the executable's, so each is checked from there.
        let mut entry_points = vec!["yue-server.exe".to_string()];
        for build in [CudaBuild::Cuda13, CudaBuild::Cuda12] {
            let backend = format!("{}/ggml-cuda.dll", build.folder());
            if bundle.join(&backend).is_file() {
                entry_points.push(backend);
            }
        }
        let handled: Vec<&str> =
            CUDA13_LIBRARIES.iter().chain(CUDA12_LIBRARIES.iter()).chain(VC_RUNTIME_LIBRARIES.iter()).copied().collect();
        for entry_point in entry_points {
            let missing = unresolved_dependencies(&bundle, &entry_point).expect("read the bundle's import tables");
            let unexpected: Vec<&String> = missing
                .iter()
                .filter(|name| !handled.iter().any(|library| library.eq_ignore_ascii_case(name)))
                .collect();
            assert!(
                unexpected.is_empty(),
                "{entry_point} imports libraries that are neither shipped nor installed on first start: {unexpected:?}. \
                 Ship them next to yue-server.exe, or add them to this module so the studio fetches them."
            );
        }
    }

    /// The import table is the whole basis of the check above; if it cannot be
    /// read, the check silently passes on anything.
    #[test]
    #[cfg(windows)]
    fn imports_are_read_out_of_a_real_binary() {
        let system = Path::new("C:/Windows/System32/notepad.exe");
        if !system.is_file() {
            return;
        }
        let imports = imported_libraries(system).expect("notepad has an import table");
        assert!(!imports.is_empty(), "no imports were read at all");
        assert!(imports.iter().all(|name| name.to_ascii_lowercase().ends_with(".dll")));
    }

    /// The libraries count as installed only once they are beside the engine.
    ///
    /// This asserts on the downloader rather than on `is_ready`, because a
    /// machine with the CUDA Toolkit is ready without downloading anything -
    /// which is the point of the PATH check, and would make the test lie about
    /// what it proved.
    #[test]
    fn the_libraries_count_as_installed_only_beside_the_engine() {
        let root = std::env::temp_dir().join(format!("engine-runtime-{}", uuid::Uuid::now_v7()));
        let runtime = EngineRuntime::new(&root);
        let asset = cublas_asset(CudaBuild::Cuda13);
        assert!(!runtime.downloader().is_installed(asset));
        assert_eq!(runtime.library_dir(), root);
        // Off CUDA the engine never loads cuBLAS, so nothing is missing.
        assert!(runtime.missing(None).is_empty());

        std::fs::create_dir_all(runtime.library_dir()).unwrap();
        for library in CUDA13_LIBRARIES {
            std::fs::write(runtime.library_dir().join(library), b"x").unwrap();
        }
        assert!(runtime.downloader().is_installed(asset));
        // The other build asks for its own cuBLAS, unless PATH has it.
        // (Linux fetches nothing: see `nothing_is_ever_downloaded_for_cublas_on_linux`.)
        if cfg!(not(target_os = "linux")) && !CUDA12_LIBRARIES.iter().all(|library| is_on_the_search_path(library)) {
            assert_eq!(runtime.missing(Some(CudaBuild::Cuda12)).len(), 1);
        }
        std::fs::remove_dir_all(&root).ok();
    }
}
