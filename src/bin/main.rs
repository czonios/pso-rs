use pso_rs::Config;
use std::env;
use std::process;

fn main() {
    let config = Config::new(env::args()).unwrap_or_else(|err| {
        eprintln!(
            "Problem parsing arguments: {}. Use -h or --help for usage information.",
            err
        );
        process::exit(1);
    });

    if let Err(e) = pso_rs::run(config) {
        eprintln!("Application error: {}", e);
        process::exit(1);
    }
}
