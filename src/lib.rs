//! pso_rs is a Particle Swarm Optimization implementation in Rust
//!
//! It uses the [`rand`](https://crates.io/crates/rand) crate for random initialization, and the [`rayon`](https://crates.io/crates/rayon) crate for parallel objective function computation.
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
//! // define a termination condition
//! fn terminate(f_best: f64) -> bool {
//!     f_best - (0.0) < 1e-4
//! }
//!
//! let config = Config {
//!     dimensions: vec![2],
//!     bounds: (-5.0, 5.0),
//!     ..Config::default()
//! };
//!
//! // define maximum number of objective function computations
//! let t_max = 10000;
//!
//! match pso_rs::run(config, objective_function) {
//!     Ok(mut pso) => {
//!         pso.run(t_max, terminate);
//!         let mut model = pso.model;
//!         println!("Model: {:?} ", model.get_f_best());
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
use pso::PSO;
use std::error::Error;

/// Creates a model and runs the PSO method
///
/// # Panics
///
/// Panics if any particle coefficient becomes NaN (usually because of bad parameterization, e.g. c1 + c2 < 4)
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
