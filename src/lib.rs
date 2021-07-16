pub mod model;

pub use model::{Config, Population};

use model::Model;
use std::error::Error;

pub fn run(config: Config) -> Result<Model, Box<dyn Error>> {
    // println!("Config: {:?}", config);

    let model = Model::new(config);
    Ok(model)
}

#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
