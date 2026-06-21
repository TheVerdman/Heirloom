use clap::{Parser, Subcommand};
use heirloom_skill::{
    build_runtime_context, compile_skill_with_options, export_sft_records, load_compiled_skill,
    run_eval, write_synthetic_tasks, write_trace_audit, CompileOptions, SkillError, SkillRegistry,
    SkillRouter,
};
use heirloom_skill::{lint_registry, util};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "heirloom-skill")]
#[command(about = "Compile and evaluate Rust-native Heirloom skill artifacts")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Compile {
        #[arg(long)]
        source: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        registry: Option<PathBuf>,
    },
    Route {
        #[arg(long)]
        skills: PathBuf,
        #[arg(long)]
        request: String,
        #[arg(long, default_value_t = 3)]
        top_k: usize,
        #[arg(long, default_value_t = false)]
        trusted_only: bool,
    },
    Context {
        #[arg(long)]
        skills: PathBuf,
        #[arg(long)]
        request: String,
        #[arg(long, default_value_t = 6000)]
        budget_chars: usize,
        #[arg(long, default_value_t = 1)]
        top_k: usize,
        #[arg(long, default_value_t = false)]
        trusted_only: bool,
    },
    SyntheticTasks {
        #[arg(long)]
        skill: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    Eval {
        #[arg(long)]
        skills: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    SftExport {
        #[arg(long)]
        traces: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = false)]
        include_failures: bool,
    },
    Inspect {
        #[arg(long)]
        skill: PathBuf,
    },
    Registry {
        #[command(subcommand)]
        command: RegistryCommands,
    },
    Trace {
        #[command(subcommand)]
        command: TraceCommands,
    },
}

#[derive(Subcommand)]
enum RegistryCommands {
    Lint {
        #[arg(long)]
        registry: PathBuf,
    },
}

#[derive(Subcommand)]
enum TraceCommands {
    Audit {
        #[arg(long)]
        traces: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

fn main() -> Result<(), SkillError> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Compile {
            source,
            out,
            registry,
        } => {
            let manifest = compile_skill_with_options(&source, &out, CompileOptions { registry })?;
            println!(
                "compiled skill {} version={} source_hash={} -> {}",
                manifest.name,
                manifest.version,
                manifest.source_hash,
                out.display()
            );
        }
        Commands::Route {
            skills,
            request,
            top_k,
            trusted_only,
        } => {
            let router = SkillRouter::from_dir_with_options(&skills, trusted_only)?;
            let matches = router.route(&request, top_k)?;
            println!("{}", util::pretty_json_string(&matches)?);
        }
        Commands::Context {
            skills,
            request,
            budget_chars,
            top_k,
            trusted_only,
        } => {
            let router = SkillRouter::from_dir_with_options(&skills, trusted_only)?;
            let matches = router.route(&request, top_k)?;
            let context = build_runtime_context(&request, &matches, &router.skills, budget_chars)?;
            print!("{}", context.rendered);
        }
        Commands::SyntheticTasks { skill, out } => {
            let skill = load_compiled_skill(&skill)?;
            let tasks = write_synthetic_tasks(&skill, &out)?;
            println!(
                "wrote synthetic tasks={} source_skill={} -> {}",
                tasks.len(),
                skill.manifest.name,
                out.display()
            );
        }
        Commands::Eval { skills, out } => {
            let report = run_eval(&skills, Some(&out))?;
            println!(
                "skill eval cases={} top1={:.3} topk={:.3} constraint_recall={:.3} -> {}",
                report.metrics.cases,
                report.metrics.top_1_routing_accuracy,
                report.metrics.top_k_routing_accuracy,
                report.metrics.constraint_recall,
                out.display()
            );
        }
        Commands::SftExport {
            traces,
            out,
            include_failures,
        } => {
            let summary = export_sft_records(&traces, &out, include_failures)?;
            println!(
                "exported sft records read={} written={} skipped_failures={} skipped_ineligible={} skipped_hygiene={} hygiene_blocking_issues={} hygiene_warning_issues={} -> {}",
                summary.read,
                summary.written,
                summary.skipped_failures,
                summary.skipped_ineligible,
                summary.skipped_hygiene,
                summary.hygiene_blocking_issues,
                summary.hygiene_warning_issues,
                out.display()
            );
        }
        Commands::Inspect { skill } => {
            let skill = load_compiled_skill(&skill)?;
            println!("{}", util::pretty_json_string(&skill.manifest)?);
        }
        Commands::Registry { command } => match command {
            RegistryCommands::Lint { registry } => {
                let registry_value = SkillRegistry::load(&registry)?;
                let report = lint_registry(&registry_value);
                println!("{}", util::pretty_json_string(&report)?);
                if !report.passed() {
                    return Err(SkillError::Invalid(format!(
                        "registry lint failed for {}",
                        registry.display()
                    )));
                }
            }
        },
        Commands::Trace { command } => match command {
            TraceCommands::Audit { traces, out } => {
                let summary = write_trace_audit(&traces, &out)?;
                println!(
                    "trace hygiene audit traces={} export_allowed={} blocking_issues={} warning_issues={} -> {}",
                    summary.traces,
                    summary.export_allowed,
                    summary.blocking_issues,
                    summary.warning_issues,
                    out.display()
                );
            }
        },
    }
    Ok(())
}
