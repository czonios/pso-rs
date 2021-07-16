pub mod model;

pub use model::Config;
type Particle = Vec<f64>;

use model::Model;
use std::error::Error;

pub fn run(config: Config) -> Result<(), Box<dyn Error>> {
    // println!("Config: {:?}", config);

    let _model = Model::new(config);
    // println!("\n\nModel: {:?} ", model);

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
