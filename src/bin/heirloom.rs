#![recursion_limit = "256"]

#[path = "heirloom/app.rs"]
mod app;

fn main() -> heirloom::Result<()> {
    app::run()
}
