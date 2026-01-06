fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Rerun the build script only if the proto file actually changes
    println!("cargo:rerun-if-changed=../../shared/proto/financial_entities.proto");

    tonic_build::configure()
        .build_server(true)   // Explicitly generate server traits
        .build_client(true)   // Explicitly generate client stubs (usually desired)
        .compile(
            &["../../shared/proto/financial_entities.proto"], // The proto file to compile
            &["../../shared/proto"],                          // The root directory for imports
        )?;

    Ok(())
}