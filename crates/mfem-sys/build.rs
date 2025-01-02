use eyre::{eyre, WrapErr};
use std::path::PathBuf;

fn main() -> eyre::Result<()> {
    let mut mfem_config = MfemConfig::detect()?;

    println!(
        "cargo:rustc-link-search=native={}",
        mfem_config.library_dir.to_str().unwrap()
    );

    for lib in mfem_config.mfem_libs {
        println!("cargo:rustc-link-lib={lib}");
    }

    mfem_config.include_dirs.push("src".into());
    let mut b = autocxx_build::Builder::new("src/lib.rs", &mfem_config.include_dirs).build()?;
    b.flag_if_supported("-std=c++14")
        .flag_if_supported("-Wno-deprecated-declarations")
        .compile("mfem-sys");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/extra.hpp");
    Ok(())
}

#[derive(Debug)]
struct MfemConfig {
    mfem_libs: Vec<String>,
    include_dirs: Vec<PathBuf>,
    library_dir: PathBuf,
    cxx_flags: Vec<String>,
}

impl MfemConfig {
    /// Find MFEM using cmake
    fn detect() -> eyre::Result<Self> {
        let version_req = semver::VersionReq::parse(">=4.6")?;

        println!("cargo:rerun-if-env-changed=MFEM_DIR");

        let dst =
            std::panic::catch_unwind(|| cmake::Config::new("MFEM").register_dep("mfem").build());
        let dst = dst.map_err(|_| {
            eyre!(
                "Pre-installed MFEM \
            not found.  You can use `vendor` feature if you do not want \
            to install MFEM system-wide."
            )
        })?;
        let cfg = std::fs::read_to_string(dst.join("share").join("mfem_info.txt"))
            .wrap_err("Something went wrong when detecting MFEM.")?;

        let mut version: Option<semver::Version> = None;
        let mut c = MfemConfig {
            mfem_libs: vec![],
            include_dirs: vec![],
            library_dir: PathBuf::new(),
            cxx_flags: vec![],
        };

        for line in cfg.lines() {
            if let Some((var, val)) = line.split_once('=') {
                match var {
                    // Keep in sync with "MFEM/CMakeLists.txt".
                    "VERSION" => version = semver::Version::parse(val).ok(),
                    "MFEM_LIBRARIES" => {
                        for l in val.split(" ") {
                            // FIXME: Right delim?
                            c.mfem_libs.push(l.into());
                        }
                    }
                    "INCLUDE_DIRS" => {
                        for d in val.split(";") {
                            c.include_dirs.push(d.into());
                        }
                    }
                    "LIBRARY_DIR" => c.library_dir = val.into(),
                    "CXX_FLAGS" => {
                        for f in val.split(" ") {
                            c.cxx_flags.push(f.into());
                        }
                    }
                    _ => (),
                }
            }
        }

        if let Some(version) = version {
            if !version_req.matches(&version) {
                panic!(
                    "Pre-installed MFEM found but version is not met \
(found {} but {} required). Please provide required version or use the \
`vendor` feature.",
                    version, version_req
                );
            }
            Ok(c)
        } else {
            panic!(
                "MFEM found but something went wrong during \
                configuration."
            );
        }
    }
}
