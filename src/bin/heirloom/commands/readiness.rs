fn run_readiness_command(command: ReadinessCommands) -> Result<()> {
    match command {
            ReadinessCommands::ValidateLearningSanity { manifest, report } => {
                let summary = validate_learning_sanity_manifest(&manifest)?;
                if let Some(report_path) = report {
                    write_json_file(&report_path, summary.clone())?;
                }
                println!(
                    "learning_sanity status={} stages={} passed={} failed={} min_loss_reduction={:.6}",
                    summary["status"].as_str().unwrap_or("unknown"),
                    summary["stages"].as_array().map(Vec::len).unwrap_or(0),
                    summary["passed"].as_u64().unwrap_or(0),
                    summary["failed"].as_u64().unwrap_or(0),
                    summary["min_loss_reduction_observed"].as_f64().unwrap_or(0.0),
                );
                if summary["status"].as_str() != Some("passed") {
                    return Err(TensorError::InvalidOperation(
                        "learning sanity validation failed".to_string(),
                    ));
                }
            }
    }
    Ok(())
}
