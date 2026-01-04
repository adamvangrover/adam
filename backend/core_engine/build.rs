fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::compile_protos("../../shared/proto/financial_entities.proto")?;
    Ok(())
}
