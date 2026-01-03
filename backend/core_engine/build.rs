fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .compile(
            &["../../shared/proto/financial_entities.proto"],
            &["../../shared/proto"],
        )?;
    Ok(())
}
