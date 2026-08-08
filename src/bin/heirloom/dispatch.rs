pub(super) fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Data { command } => run_data_command(command)?,
        Commands::Tokenizer { command } => run_tokenizer_command(command)?,
        Commands::Gpu { command } => run_gpu_command(command)?,
        Commands::Readiness { command } => run_readiness_command(command)?,
        Commands::Padawan { command } => run_padawan_command(command)?,
        command @ (Commands::TrainLm { .. } | Commands::TrainMemoryLm { .. }) => {
            run_training_command(command)?;
        }
        command
        @ (Commands::TrainLmRank { .. }
        | Commands::TrainMemoryLmRank { .. }
        | Commands::NcclProbeRank { .. }
        | Commands::NcclUniqueId { .. }
        | Commands::LauncherTest { .. }
        | Commands::LauncherTestRank { .. }) => {
            run_internal_command(command)?;
        }
        command
        @ (Commands::Generate { .. }
        | Commands::GenerateMemoryLm { .. }
        | Commands::EvalLm { .. }
        | Commands::EvalMemoryLm { .. }) => {
            run_inference_command(command)?;
        }
    }
    Ok(())
}
