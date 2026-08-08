fn run_internal_command(command: Commands) -> Result<()> {
    match command {
        Commands::TrainLmRank { config } => {
            run_train_lm_rank(&config)?;
        }
        Commands::TrainMemoryLmRank { config } => {
            run_train_memory_lm_rank(&config)?;
        }
        Commands::NcclProbeRank { config } => {
            run_nccl_probe_rank(&config)?;
        }
        Commands::NcclUniqueId { hold_secs } => {
            let root = cuda::NcclUniqueIdRoot::new().map_err(cuda_error)?;
            println!(
                "{}{}",
                NCCL_UNIQUE_ID_HELPER_MARKER,
                root.unique_id().to_hex()
            );
            std::io::stdout().flush().map_err(|err| {
                TensorError::Io(format!(
                    "failed to flush NCCL unique-id helper stdout: {err}"
                ))
            })?;
            if let Some(seconds) = hold_secs {
                eprintln!("NCCL unique-id helper holding bootstrap endpoint for {seconds}s");
                std::thread::sleep(Duration::from_secs(seconds));
            }
        }
        Commands::LauncherTest {
            ranks,
            fail_rank,
            hang_rank,
            timeout_secs,
            rank_start_timeout_secs,
            kill_grace_secs,
            report,
        } => {
            run_launcher_test(LauncherTestConfig {
                ranks,
                fail_rank,
                hang_rank,
                timeout: Duration::from_secs(timeout_secs),
                rank_start_timeout: Duration::from_secs(rank_start_timeout_secs),
                kill_grace: Duration::from_secs(kill_grace_secs),
                report,
            })?;
        }
        Commands::LauncherTestRank { config } => {
            run_launcher_test_rank(&config)?;
        }
        _ => unreachable!("internal dispatcher received a public command"),
    }
    Ok(())
}
