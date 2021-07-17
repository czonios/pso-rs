//! An easy-to-use, simple Particle Swarm Optimization (PSO) implementation in Rust
//!
//! ![Crates.io](https://img.shields.io/crates/v/pso_rs)
//! ![docs.rs](https://img.shields.io/docsrs/pso-rs)
//!
//! It uses the [`rand`](https://crates.io/crates/rand) crate for random initialization, and the [`rayon`](https://crates.io/crates/rayon) crate for parallel objective function computation.
//!
//! The [example](#examples) below can get you started.
//! In order to use it on your own optimization problem, you will need to define an objective function as it is defined in the [run](fn.run.html) function, and a [`Config`](model/struct.Config.html) object. See the [Notes](#notes) section for more tips.
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
//!
//! # Notes
//!
//! Even though you can have particles of any shape and size, as long as each item is `f64`, `pso_rs` represents each particle as a flat vector: `Vec<f64>`.
//!
//! This means that, for example, in order to find clusters of 20 molecules in 3D space that minimize the [Lennard-Jones potential energy](https://en.wikipedia.org/wiki/Lennard-Jones_potential), you can define `dimensions` as (20, 3):
//! ```rust
//! let config = Config {
//!     dimensions: vec![20, 3],
//!     bounds: (-5.0, 5.0),
//!     ..Config::default()
//! };
//! ```
//!
//! If you want, you can also create a custom `reshape` function, like this one for molecule clusters below:
//!
//! ```rust
//! fn reshape(particle: &Particle, particle_dims: &Vec<usize>) -> Vec<Vec<f64>> {
//!     let mut reshaped_population = vec![];
//!     let mut i = 0;
//!     for _ in 0..particle_dims[0] {
//!         let mut reshaped_particle = vec![];
//!         for _ in 0..particle_dims[1] {
//!             reshaped_particle.push(particle[i]);
//!             i += 1;
//!         }
//!         reshaped_population.push(reshaped_particle);
//!     }
//!     reshaped_population
//! }
//!```
//!
//! Then you can use that to reshape the particle at any point, for example to print the minimizer or to use in the objective function you have defined:
//!
//! ```rust
//! // somewhere in main(), after running PSO as in the example:
//! println!(
//!     "Best found minimizer: {:#?} ",
//!     reshape(&model.get_x_best(), &model.config.dimensions)
//! );
//!
//! // used in the objective function
//! fn objective_function(p: &Particle, flat_dim: usize, dimensions: &Vec<usize>) -> f64 {
//!      let reshaped_particle = reshape(p, dimensions);
//!     /* Do stuff */
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
