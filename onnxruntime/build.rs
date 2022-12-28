use std::env;

fn main() {
    let debug = env::var("DEBUG").unwrap().parse().unwrap();
    if debug || boolean_env_var("ORT_ALLOW_VERBOSE_LOGGING_ON_RELEASE") {
        println!("cargo:rustc-cfg=allow_verbose_logging");
    }
    println!("cargo:rerun-if-env-changed=ORT_ALLOW_VERBOSE_LOGGING_ON_RELEASE");
}

fn boolean_env_var(name: &str) -> bool {
    // Same as `ORT_USE_CUDA`.
    let var = env::var(name).unwrap_or_default();
    matches!(&*var.to_lowercase(), "1" | "yes" | "true" | "on")
}
