#![allow(dead_code)]

use std::{
    borrow::Cow,
    env, fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    str::FromStr,
};

/// ONNX Runtime version
///
/// WARNING: If version is changed, bindings for all platforms will have to be re-generated.
///          To do so, run this:
///              cargo build --package onnxruntime-sys --features generate-bindings
const ORT_VERSION: &str = "1.14.0";

/// Base Url from which to download pre-built releases/
const ORT_RELEASE_BASE_URL: &str = "https://github.com/microsoft/onnxruntime/releases/download";

/// Base Url from which to download pre-built releases for android/
const ORT_MAVEN_RELEASE_BASE_URL: &str =
    "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android";

/// Base Url from which to download ios pre-build releases/
const ORT_IOS_RELEASE_BASE_URL: &str =
    "https://github.com/VOICEVOX/onnxruntime-builder/releases/download";

/// onnxruntime repository/
const ORT_REPOSITORY_URL: &str = "https://github.com/microsoft/onnxruntime.git";

/// Environment variable selecting which strategy to use for finding the library
/// Possibilities:
/// * "download": Download a pre-built library from upstream. This is the default if `ORT_STRATEGY` is not set.
/// * "system": Use installed library. Use `ORT_LIB_LOCATION` to point to proper location.
/// * "compile": Download source and compile (TODO).
const ORT_ENV_STRATEGY: &str = "ORT_STRATEGY";

/// Name of environment variable that, if present, contains the location of a pre-built library.
/// Only used if `ORT_STRATEGY=system`.
const ORT_ENV_SYSTEM_LIB_LOCATION: &str = "ORT_LIB_LOCATION";
/// Name of environment variable that, if present, controls wether to use CUDA or not.
const ORT_ENV_GPU: &str = "ORT_USE_CUDA";

/// onnxrutnime ライブラリを出力するためのディレクトリ
const ORT_ENV_OUT_DIR: &str = "ORT_OUT_DIR";

/// Subdirectory (of the 'target' directory) into which to extract the prebuilt library.
const ORT_PREBUILT_EXTRACT_DIR: &str = "onnxruntime";

static TRIPLET: once_cell::sync::Lazy<Triplet> = once_cell::sync::Lazy::new(|| Triplet {
    os: env::var("CARGO_CFG_TARGET_OS")
        .expect("Unable to get TARGET_OS")
        .parse()
        .unwrap(),
    arch: env::var("CARGO_CFG_TARGET_ARCH")
        .expect("Unable to get TARGET_ARCH")
        .parse()
        .unwrap(),
    accelerator: env::var(ORT_ENV_GPU).unwrap_or_default().parse().unwrap(),
});

static ONNXRUNTIME_DIR_NAME: once_cell::sync::Lazy<String> =
    once_cell::sync::Lazy::new(
        || format!("onnxruntime-{}-{}", TRIPLET.as_onnx_str(), ORT_VERSION,),
    );

#[cfg(feature = "disable-sys-build-script")]
fn main() {
    println!("Build script disabled!");
}

#[cfg(not(feature = "disable-sys-build-script"))]
fn main() {
    let libort_install_dir = prepare_libort_dir();

    // FIXME: directmlとandroidで処理の表現が違うので統一する
    #[cfg(not(any(feature = "directml", feature = "coreml")))]
    let (include_dir, lib_dir) = match TRIPLET.os {
        Os::Android => {
            let include_dir = libort_install_dir.join("headers");
            let runtimes_dir = libort_install_dir
                .join("jni")
                .join(&*TRIPLET.arch.as_onnx_android_str());
            (include_dir, runtimes_dir)
        }
        _ => (
            libort_install_dir.join("include"),
            libort_install_dir.join("lib"),
        ),
    };

    #[cfg(feature = "directml")]
    let (include_dir, lib_dir) = {
        let include_dir = libort_install_dir.join("build/native/include");
        let runtimes_dir = libort_install_dir
            .join("runtimes")
            .join(format!(
                "{}-{}",
                TRIPLET.os.as_onnx_str(),
                TRIPLET.arch.as_onnx_directml_str()
            ))
            .join("native");

        let export_libort_dir = libort_install_dir.join(&*ONNXRUNTIME_DIR_NAME);
        let export_include_dir = export_libort_dir.join("include");
        let export_lib_dir = export_libort_dir.join("lib");
        fs::create_dir_all(&export_include_dir).unwrap();
        copy_all_files(include_dir, &export_include_dir);

        fs::create_dir_all(&export_lib_dir).unwrap();
        copy_all_files(runtimes_dir, &export_lib_dir);
        (export_include_dir, export_lib_dir)
    };

    #[cfg(feature = "coreml")]
    let (include_dir, lib_dir) = {
        let include_dir = libort_install_dir.join("build/native/include");
        let runtimes_dir = libort_install_dir
            .join("runtimes")
            .join(format!(
                "{}-{}",
                TRIPLET.os.as_onnx_str(),
                TRIPLET.arch.as_onnx_directml_str()
            ))
            .join("native");

        let export_libort_dir = libort_install_dir.join(&*ONNXRUNTIME_DIR_NAME);
        let export_include_dir = export_libort_dir.join("include");
        let export_lib_dir = export_libort_dir.join("lib");
        fs::create_dir_all(&export_include_dir).unwrap();
        copy_all_files(include_dir, &export_include_dir);

        fs::create_dir_all(&export_lib_dir).unwrap();
        copy_all_files(runtimes_dir, &export_lib_dir);
        (export_include_dir, export_lib_dir)
    };
    if let Ok(ort_lib_out_dir) = env::var(ORT_ENV_OUT_DIR) {
        output_onnxruntime_library(&lib_dir, &ort_lib_out_dir);
        for entry in lib_dir.read_dir().unwrap().flat_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_file() {
                let file_name = path.file_name().unwrap().to_str().unwrap();
                println!(
                    "cargo:rerun-if-changed={}",
                    Path::new(&ort_lib_out_dir).join(file_name).display()
                );
            }
        }
    }

    println!("Include directory: {:?}", include_dir);
    println!("Lib directory: {:?}", lib_dir);

    // Tell cargo to tell rustc to link onnxruntime shared library.
    println!("cargo:rustc-link-lib=onnxruntime");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    println!("cargo:rerun-if-env-changed={}", ORT_ENV_STRATEGY);
    println!("cargo:rerun-if-env-changed={}", ORT_ENV_GPU);
    println!("cargo:rerun-if-env-changed={}", ORT_ENV_SYSTEM_LIB_LOCATION);
    println!("cargo:rerun-if-env-changed={}", ORT_ENV_OUT_DIR);

    generate_bindings(&include_dir);
}

fn output_onnxruntime_library(ort_lib_dir: impl AsRef<Path>, ort_lib_out_dir: impl AsRef<Path>) {
    let ort_lib_dir = ort_lib_dir.as_ref();
    let ort_lib_out_dir = ort_lib_out_dir.as_ref();
    fs::create_dir_all(ort_lib_out_dir).unwrap();

    for entry in ort_lib_dir.read_dir().unwrap().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file() {
            fs::copy(
                &path,
                ort_lib_out_dir.join(path.file_name().unwrap().to_str().unwrap()),
            )
            .unwrap();
        }
    }
}

#[cfg(not(feature = "generate-bindings"))]
fn generate_bindings(_include_dir: &Path) {
    println!("Bindings not generated automatically, using committed files instead.");
    println!("Enable with the 'generate-bindings' cargo feature.");

    // NOTE: If bindings could not be be generated for Apple Sillicon M1, please uncomment the following
    // let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
    // let arch = env::var("CARGO_CFG_TARGET_ARCH").expect("Unable to get TARGET_ARCH");
    // if os == "macos" && arch == "aarch64" {
    //     panic!(
    //         "OnnxRuntime {} bindings for Apple M1 are not available",
    //         ORT_VERSION
    //     );
    // }
}

fn copy_all_files(from: impl AsRef<Path>, to: impl AsRef<Path>) {
    let from = from.as_ref();
    let to = to.as_ref();
    for entry in fs::read_dir(from).unwrap().filter_map(|e| e.ok()) {
        let path = entry.path();
        if !path.is_dir() {
            if let Some(file_name) = path.file_name() {
                let dest_path = to.join(file_name);
                fs::copy(path, dest_path).unwrap();
            }
        }
    }
}

#[cfg(feature = "generate-bindings")]
fn generate_bindings(include_dir: &Path) {
    let clang_args = &[
        format!("-I{}", include_dir.display()),
        format!(
            "-I{}",
            include_dir
                .join("onnxruntime")
                .join("core")
                .join("session")
                .display()
        ),
        #[cfg(feature = "directml")]
        format!("-D{}", "USE_DML"),
    ];

    #[cfg(not(feature = "directml"))]
    let header_name = "wrapper.h";
    #[cfg(feature = "directml")]
    let header_name = "wrapper_directml.h";

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed={}", header_name);
    println!("cargo:rerun-if-changed=src/generated/bindings.rs");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let mut bind_builder = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header(header_name)
        // The current working directory is 'onnxruntime-sys'
        .clang_args(clang_args)
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Set `size_t` to be translated to `usize` for win32 compatibility.
        .size_t_is_usize(true)
        // Format using rustfmt
        .rustfmt_bindings(true)
        .rustified_enum("*");

    for entry in include_dir.read_dir().unwrap().filter_map(|e| e.ok()) {
        let path = entry.path();
        let file_name = path.file_name().unwrap().to_str().unwrap().to_string();
        bind_builder =
            bind_builder.allowlist_file(format!(".*{}", file_name.replace(".h", "\\.h")));
    }
    let bindings = bind_builder
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to (source controlled) src/generated/<os>/<arch>/bindings.rs
    let generated_file = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("src")
        .join("generated")
        .join(env::var("CARGO_CFG_TARGET_OS").unwrap())
        .join(env::var("CARGO_CFG_TARGET_ARCH").unwrap());
    #[cfg(not(any(feature = "directml", feature = "coreml")))]
    let generated_file = generated_file.join("bindings.rs");
    #[cfg(feature = "directml")]
    let generated_file = generated_file.join("bindings_directml.rs");
    #[cfg(feature = "coreml")]
    let generated_file = generated_file.join("bindings_coreml.rs");
    println!("cargo:rerun-if-changed={:?}", generated_file);
    bindings
        .write_to_file(&generated_file)
        .expect("Couldn't write bindings!");
}

fn download<P>(source_url: &str, target_file: P)
where
    P: AsRef<Path>,
{
    let resp = ureq::get(source_url)
        .timeout(std::time::Duration::from_secs(300))
        .call()
        .unwrap_or_else(|err| panic!("ERROR: Failed to download {}: {:?}", source_url, err));

    let len = resp
        .header("Content-Length")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap();
    let mut reader = resp.into_reader();
    // FIXME: Save directly to the file
    let mut buffer = vec![];
    let read_len = reader.read_to_end(&mut buffer).unwrap();
    assert_eq!(buffer.len(), len);
    assert_eq!(buffer.len(), read_len);

    let f = fs::File::create(&target_file).unwrap();
    let mut writer = io::BufWriter::new(f);
    writer.write_all(&buffer).unwrap();
}

fn extract_archive(filename: &Path, output: &Path) {
    match filename.extension().map(|e| e.to_str()) {
        Some(Some("zip")) => extract_zip(filename, output),
        Some(Some("tgz")) => extract_tgz(filename, output),
        Some(Some("aar")) => extract_zip(filename, output),
        _ => unimplemented!(),
    }
}

fn extract_tgz(filename: &Path, output: &Path) {
    let file = fs::File::open(filename).unwrap();
    let buf = io::BufReader::new(file);
    let tar = flate2::read::GzDecoder::new(buf);
    let mut archive = tar::Archive::new(tar);
    archive.unpack(output).unwrap();
}

fn extract_zip(filename: &Path, outpath: &Path) {
    let file = fs::File::open(filename).unwrap();
    let buf = io::BufReader::new(file);
    let mut archive = zip::ZipArchive::new(buf).unwrap();
    for i in 0..archive.len() {
        let mut file = archive.by_index(i).unwrap();
        #[allow(deprecated)]
        let outpath = outpath.join(file.sanitized_name());
        if !file.name().ends_with('/') {
            println!(
                "File {} extracted to \"{}\" ({} bytes)",
                i,
                outpath.as_path().display(),
                file.size()
            );
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    fs::create_dir_all(p).unwrap();
                }
            }
            let mut outfile = fs::File::create(&outpath).unwrap();
            io::copy(&mut file, &mut outfile).unwrap();
        }
    }
}

trait OnnxPrebuiltArchive {
    fn as_onnx_str(&self) -> Cow<str>;
}

#[derive(Debug)]
enum Architecture {
    X86,
    X86_64,
    Arm,
    Arm64,
}

impl FromStr for Architecture {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "x86" => Ok(Architecture::X86),
            "x86_64" => Ok(Architecture::X86_64),
            "arm" => Ok(Architecture::Arm),
            "aarch64" => Ok(Architecture::Arm64),
            _ => Err(format!("Unsupported architecture: {}", s)),
        }
    }
}

impl OnnxPrebuiltArchive for Architecture {
    fn as_onnx_str(&self) -> Cow<str> {
        match self {
            Architecture::X86 => Cow::from("x86"),
            Architecture::X86_64 => Cow::from("x86_64"),
            Architecture::Arm => Cow::from("arm"),
            Architecture::Arm64 => Cow::from("aarch64"),
        }
    }
}

impl Architecture {
    fn as_onnx_directml_str(&self) -> Cow<str> {
        match self {
            Architecture::X86 => Cow::from("x86"),
            Architecture::X86_64 => Cow::from("x64"),
            Architecture::Arm => Cow::from("arm"),
            Architecture::Arm64 => Cow::from("arm64"),
        }
    }

    fn as_onnx_android_str(&self) -> Cow<str> {
        match self {
            Architecture::X86 => Cow::from("x86"),
            Architecture::X86_64 => Cow::from("x86_64"),
            Architecture::Arm => Cow::from("armeabi-v7a"),
            Architecture::Arm64 => Cow::from("arm64-v8a"),
        }
    }
}

#[derive(Debug)]
#[allow(clippy::enum_variant_names)]
enum Os {
    Windows,
    Linux,
    MacOs,
    Android,
    IOs,
}

impl Os {
    fn archive_extension(&self) -> &'static str {
        match self {
            Os::Windows => "zip",
            Os::Linux => "tgz",
            Os::MacOs => "tgz",
            Os::Android => "aar",
            Os::IOs => "tgz",
        }
    }
}

impl FromStr for Os {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "windows" => Ok(Os::Windows),
            "macos" => Ok(Os::MacOs),
            "linux" => Ok(Os::Linux),
            "android" => Ok(Os::Android),
            "ios" => Ok(Os::IOs),
            _ => Err(format!("Unsupported os: {}", s)),
        }
    }
}

impl OnnxPrebuiltArchive for Os {
    fn as_onnx_str(&self) -> Cow<str> {
        match self {
            Os::Windows => Cow::from("win"),
            Os::Linux => Cow::from("linux"),
            Os::MacOs => Cow::from("osx"),
            Os::Android => Cow::from("android"),
            Os::IOs => Cow::from("ios"),
        }
    }
}

#[derive(Debug)]
enum Accelerator {
    None,
    Gpu,
}

impl FromStr for Accelerator {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "1" | "yes" | "true" | "on" => Ok(Accelerator::Gpu),
            _ => Ok(Accelerator::None),
        }
    }
}

impl OnnxPrebuiltArchive for Accelerator {
    fn as_onnx_str(&self) -> Cow<str> {
        match self {
            Accelerator::None => Cow::from(""),
            Accelerator::Gpu => Cow::from("gpu"),
        }
    }
}

#[derive(Debug)]
struct Triplet {
    os: Os,
    arch: Architecture,
    accelerator: Accelerator,
}

impl OnnxPrebuiltArchive for Triplet {
    fn as_onnx_str(&self) -> Cow<str> {
        match (&self.os, &self.arch, &self.accelerator) {
            (Os::Windows, Architecture::X86_64, Accelerator::None)
            | (Os::Windows, Architecture::Arm, Accelerator::None)
            | (Os::Windows, Architecture::Arm64, Accelerator::None)
            | (Os::Linux, Architecture::X86_64, Accelerator::None) => {
                Cow::from(format!("{}-{}", self.os.as_onnx_str(), "x64"))
            }
            (Os::Windows, Architecture::X86, Accelerator::None)
            | (Os::MacOs, Architecture::X86_64, Accelerator::None)
            | (Os::Android, Architecture::X86_64, Accelerator::None)
            | (Os::Android, Architecture::Arm64, Accelerator::None)
            | (Os::Linux, Architecture::Arm64, Accelerator::None) => Cow::from(format!(
                "{}-{}",
                self.os.as_onnx_str(),
                self.arch.as_onnx_str()
            )),
            (Os::MacOs, Architecture::Arm64, Accelerator::None) => {
                Cow::from(format!("{}-{}", self.os.as_onnx_str(), "arm64"))
            }
            // onnxruntime-win-gpu-x64-1.8.1.zip
            // Note how this one is inverted from the linux one next
            (Os::Windows, Architecture::X86_64, Accelerator::Gpu) => Cow::from(format!(
                "{}-{}-{}",
                self.os.as_onnx_str(),
                "x64",
                self.accelerator.as_onnx_str(),
            )),
            // onnxruntime-linux-x64-gpu-1.8.1.tgz
            // Note how this one is inverted from the windows one above
            (Os::Linux, Architecture::X86_64, Accelerator::Gpu) => Cow::from(format!(
                "{}-{}-{}",
                self.os.as_onnx_str(),
                "x64",
                self.accelerator.as_onnx_str(),
            )),
            // onnxruntime-ios-arm64-1.8.1.tgz
            // Note aarch64 simulator have a '-sim' in the target, but x86_64 does not; both have '-sim' for onnxruntime
            (Os::IOs, Architecture::Arm64, Accelerator::None) => {
                let os = if env::var("TARGET").unwrap().ends_with("sim") {
                    format!("{}-sim", self.os.as_onnx_str())
                } else {
                    format!("{}", self.os.as_onnx_str())
                };
                Cow::from(format!("{}-{}", os, "arm64"))
            }
            // onnxruntime-ios-sim-x86_64-1.8.1.tgz
            (Os::IOs, Architecture::X86_64, Accelerator::None) => Cow::from(format!(
                "{}-sim-{}",
                self.os.as_onnx_str(),
                self.arch.as_onnx_str()
            )),
            _ => {
                panic!(
                    "Unsupported prebuilt triplet: {:?}, {:?}, {:?}. Please use {}=system and {}=/path/to/onnxruntime",
                    self.os, self.arch, self.accelerator, ORT_ENV_STRATEGY, ORT_ENV_SYSTEM_LIB_LOCATION
                );
            }
        }
    }
}

fn prebuilt_archive_url() -> (PathBuf, String) {
    #[cfg(not(feature = "directml"))]
    let prebuilt_archive = format!(
        "{}.{}",
        &*ONNXRUNTIME_DIR_NAME,
        TRIPLET.os.archive_extension()
    );

    #[cfg(feature = "directml")]
    let prebuilt_archive = format!(
        "Microsoft.ML.OnnxRuntime.DirectML.{}.{}",
        ORT_VERSION,
        TRIPLET.os.archive_extension()
    );
    let prebuilt_url = match TRIPLET.os {
        Os::Android => format!(
            "{}/{}/onnxruntime-android-{}.{}",
            ORT_MAVEN_RELEASE_BASE_URL,
            ORT_VERSION,
            ORT_VERSION,
            TRIPLET.os.archive_extension()
        ),
        Os::IOs => format!(
            "{}/{}/{}",
            ORT_IOS_RELEASE_BASE_URL, ORT_VERSION, prebuilt_archive
        ),
        _ => format!(
            "{}/v{}/{}",
            ORT_RELEASE_BASE_URL, ORT_VERSION, prebuilt_archive
        ),
    };
    (PathBuf::from(prebuilt_archive), prebuilt_url)
}

fn prepare_libort_dir_prebuilt() -> PathBuf {
    let (prebuilt_archive, prebuilt_url) = prebuilt_archive_url();

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let extract_dir = out_dir.join(format!(
        "{}_{}_{}",
        ORT_PREBUILT_EXTRACT_DIR,
        ORT_VERSION,
        if matches!(TRIPLET.accelerator, Accelerator::Gpu) {
            "gpu"
        } else {
            #[cfg(not(feature = "directml"))]
            let accelerator = "cpu";
            #[cfg(feature = "directml")]
            let accelerator = "directml";
            accelerator
        }
    ));
    let downloaded_file = out_dir.join(&prebuilt_archive);

    println!("cargo:rerun-if-changed={}", downloaded_file.display());

    if !downloaded_file.exists() {
        println!("Creating directory {:?}", out_dir);
        fs::create_dir_all(&out_dir).unwrap();

        println!(
            "Downloading {} into {}",
            prebuilt_url,
            downloaded_file.display()
        );
        download(&prebuilt_url, &downloaded_file);
    }

    if !extract_dir.exists() {
        println!("Extracting to {}...", extract_dir.display());
        extract_archive(&downloaded_file, &extract_dir);
    }

    // directmlの場合はzipの子ディレクトリがzipファイル名のディレクトリではないため、
    // この処理は非directmlの場合のみ行う
    #[cfg(not(feature = "directml"))]
    let extract_dir = match TRIPLET.os {
        Os::Android => extract_dir,
        _ => extract_dir.join(prebuilt_archive.file_stem().unwrap()),
    };

    extract_dir
}

fn prepare_libort_dir() -> PathBuf {
    let strategy = env::var(ORT_ENV_STRATEGY);
    println!(
        "strategy: {:?}",
        strategy
            .as_ref()
            .map(String::as_str)
            .unwrap_or_else(|_| "unknown")
    );
    match strategy.as_ref().map(String::as_str) {
        Ok("download") | Err(_) => prepare_libort_dir_prebuilt(),
        Ok("system") => PathBuf::from(match env::var(ORT_ENV_SYSTEM_LIB_LOCATION) {
            Ok(p) => p,
            Err(e) => {
                panic!(
                    "Could not get value of environment variable {:?}: {:?}",
                    ORT_ENV_SYSTEM_LIB_LOCATION, e
                );
            }
        }),
        Ok("compile") => unimplemented!(),
        _ => panic!("Unknown value for {:?}", ORT_ENV_STRATEGY),
    }
}
