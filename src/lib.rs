//! pso_rs is a PSO implementation in Rust
//!
//! The [example](#examples) below can get you started.
//! In order to use it on your own optimization problem, you will need to define an objective function as it is defined in the [run](fn.run.html) function, and a [`Config`](model/struct.Config.html) object.
//!
//! # Examples
//!
//! ```rust
//! use pso_rs::model::*;
//!
//! // define objective function (Rosenbrock)
//! fn objective_function(p: &Particle, _flat_dim: usize, _dimensions: &Vec<usize>) -> f64 {
//!     // x = p[0], y = p[1]
//!     (1.0-p[0]).powf(2.0) + 100.0 * ((p[1]-p[0]).powf(2.0)).powf(2.0)
//! }
//!
//! let config = Config {
//!     dimensions: vec![2],
//!     bounds: (-5.0, 5.0),
//!     ..Config::default()
//! };
//!     
//!     match pso_rs::run(config, objective_function) {
//!     Ok(mut pso) => {
//!         pso.run(1000, |f_best| {
//!             f_best - (0.0) < 1e-4
//!         });
//!         let mut model = pso.model;
//!         println!("Model: {:?} ", model.get_error());
//!     }
//!     Err(e) => {
//!         eprintln!("Could not construct PSO: {}", e);
//!     }
//! }
//! ```

pub mod model;
mod pso;

pub use model::{Config, NeighborhoodType, Particle, Population};

use model::Model;
pub use pso::PSO;
use std::error::Error;

pub fn run(
    config: Config,
    obj_f: fn(&Particle, usize, &Vec<usize>) -> f64,
) -> Result<PSO, Box<dyn Error>> {
    let model = Model::new(config, obj_f);
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
