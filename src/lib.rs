pub mod model;
mod pso;

pub use model::{Config, Population};

use model::Model;
use pso::PSO;
use std::error::Error;

pub fn run(config: Config) -> Result<PSO, Box<dyn Error>> {
    // println!("Config: {:?}", config);

    let model = Model::new(config);
    let pso = PSO::new(model);
    Ok(pso)
}

#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
