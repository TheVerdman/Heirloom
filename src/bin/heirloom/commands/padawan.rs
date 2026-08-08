fn run_padawan_command(command: PadawanCommands) -> Result<()> {
    match command {
            PadawanCommands::Validate {
                episodes,
                artifact_root,
                report,
            } => {
                let summary = padawan_validate_report(&episodes, artifact_root.as_deref())?;
                if let Some(report_path) = report {
                    write_json_file(&report_path, summary.clone())?;
                }
                println!(
                    "validated padawan episodes={} sft_eligible={} smft_eligible={} max_guidance_level={}",
                    summary["episodes"].as_u64().unwrap_or(0),
                    summary["sft_eligible"].as_u64().unwrap_or(0),
                    summary["smft_eligible"].as_u64().unwrap_or(0),
                    summary["max_guidance_level"].as_u64().unwrap_or(0),
                );
            }
            PadawanCommands::Verify {
                episodes,
                artifact_root,
                family,
                allow_path_prefix,
                report,
            } => {
                let summary = padawan_verify_report(
                    &episodes,
                    artifact_root.as_deref(),
                    &family,
                    &allow_path_prefix,
                )?;
                if let Some(report_path) = report {
                    write_json_file(&report_path, summary.clone())?;
                }
                println!(
                    "padawan verifier harness status={} episodes={} families={} passed={} failed={} skipped={}",
                    summary["status"].as_str().unwrap_or("unknown"),
                    summary["episodes"].as_array().map(Vec::len).unwrap_or(0),
                    summary["families_requested"].as_array().map(Vec::len).unwrap_or(0),
                    summary["family_counts"]["passed"].as_u64().unwrap_or(0),
                    summary["family_counts"]["failed"].as_u64().unwrap_or(0),
                    summary["family_counts"]["skipped"].as_u64().unwrap_or(0),
                );
                if summary["status"].as_str() != Some("passed") {
                    return Err(TensorError::InvalidOperation(
                        "padawan verifier harness failed".to_string(),
                    ));
                }
            }
    }
    Ok(())
}
