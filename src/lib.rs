//! An easy-to-use, simple Particle Swarm Optimization (PSO) implementation in Rust.
//!
//! [![Crates.io](https://img.shields.io/crates/v/pso_rs?style=for-the-badge)](https://crates.io/crates/pso-rs)
//! [![docs.rs](https://img.shields.io/docsrs/pso-rs?style=for-the-badge)](https://docs.rs/pso-rs/latest/pso_rs/)
//! [![GitHub](https://img.shields.io/github/license/czonios/pso-rs?style=for-the-badge)](https://github.com/czonios/pso-rs/blob/master/LICENSE)
//!
//! It uses the [`rand`](https://crates.io/crates/rand) crate for random initialization, and the [`rayon`](https://crates.io/crates/rayon) crate for parallel objective function computation. It also has a nice progress bar curtesy of the [`indicatif`](https://crates.io/crates/indicatif) crate.
//!
//! The [example](#examples) below can get you started.
//! In order to use it on your own optimization problem, you will need to define an objective function as it is defined in the [run](fn.run.html) function, and a [`Config`](model/struct.Config.html) object. See the [Notes](#notes) section for more tips.
//!
//! # Examples
//!
//! ## Run PSO
//!
//! ```rust
//! use pso_rs::*;
//!
//! // define objective function (d-dimensional Rosenbrock)
//! fn objective_function(
//!     p: &Particle,
//!     _flat_dim: usize,
//!     dimensions: &Vec<usize>
//! ) -> f64 {
//!     (0..dimensions[0] - 1).map(|i| {
//!         100.0 * ((p[i+1]-p[i]).powf(2.0)).powf(2.0)
//!             + (1.0-p[i]).powf(2.0)
//!     }).sum()
//! }
//!
//! // define a termination condition (optional)
//! fn terminate(f_best: f64) -> bool {
//!     f_best - (0.0) < 1e-4
//! }
//!
//! let config = Config {
//!     // dimension shape of each particle
//!     dimensions: vec![2],
//!     // problem bounds in each dimension
//!     bounds: (-5.0, 10.0),
//!     // maximum no. of objective function computations
//!     t_max: 10000,
//!     // leave the rest of the params as default
//!     ..Config::default()
//! };
//!
//! let pso = pso_rs::run(
//!     config,
//!     objective_function,
//!     Some(terminate)
//! ).unwrap();
//!     
//! let model = pso.model;
//! println!("Model: {:?} ", model.get_f_best());
//! ```
//!
//! ## Initialize PSO for later execution
//!
//! ```rust
//! use pso_rs::*;
//!
//! // define objective function (d-dimensional Rosenbrock)
//! fn objective_function(
//!     p: &Particle,
//!     _flat_dim: usize,
//!     dimensions: &Vec<usize>
//! ) -> f64 {
//!     (0..dimensions[0] - 1).map(|i| {
//!         100.0 * ((p[i+1]-p[i]).powf(2.0)).powf(2.0)
//!             + (1.0-p[i]).powf(2.0)
//!     }).sum()
//! }
//!
//!
//! let config = Config {
//!     dimensions: vec![2],
//!     bounds: (-5.0, 10.0),
//!     t_max: 10000,
//!     ..Config::default()
//! };
//!
//! let mut pso = pso_rs::init(
//!     config,
//!     objective_function
//! ).unwrap();
//!
//! // run PSO with no termination condition
//! pso.run(|_| false);
//!     
//! let model = pso.model;
//! println!("Model: {:?} ", model.get_f_best());
//! ```
//!
//! # Notes
//!
//! Even though you can have particles of any shape and size, as long as each item is `f64`, `pso_rs` represents each particle as a flat vector: `Vec<f64>`.
//!
//! This means that, for example, in order to find clusters of 20 molecules in 3D space that minimize the [Lennard-Jones potential energy](https://en.wikipedia.org/wiki/Lennard-Jones_potential), you can define `dimensions` as (20, 3).
//! If you want, you can also create a custom `reshape` function, like this one for molecule clusters below:
//!
//! ```rust
//! use pso_rs::*;
//!
//! fn reshape(
//!     particle: &Particle,
//!     particle_dims: &Vec<usize>
//! ) -> Vec<Vec<f64>> {
//!     let mut reshaped_cluster = vec![];
//!     let mut i = 0;
//!     for _ in 0..particle_dims[0] {
//!         let mut reshaped_molecule = vec![];
//!         for _ in 0..particle_dims[1] {
//!             reshaped_molecule.push(particle[i]);
//!             i += 1;
//!         }
//!         reshaped_cluster.push(reshaped_molecule);
//!     }
//!     reshaped_cluster
//! }
//!
//! // used in the objective function
//! fn objective_function(
//!     p: &Particle,
//!     _flat_dim: usize,
//!     dimensions: &Vec<usize>
//! ) -> f64 {
//!     let _reshaped_particle = reshape(p, dimensions);
//!     /* Do stuff */
//!     0.0
//! }
//!
//! let config = Config {
//!     dimensions: vec![20, 3],
//!     bounds: (-2.5, 2.5),
//!     t_max: 1,
//!     ..Config::default()
//! };
//!
//! let pso = pso_rs::run(
//!     config,
//!     objective_function,
//!     None
//! ).unwrap();
//!
//! // somewhere in main(), after running PSO as in the example:
//! println!(
//!     "Best found minimizer: {:#?} ",
//!     reshape(&pso.model.get_x_best(),
//!         &pso.model.config.dimensions)
//! );
//! ```

pub mod model;
mod pso;

pub use model::*;

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
    terminate_f: Option<fn(f64) -> bool>,
) -> Result<PSO, Box<dyn Error>> {
    let mut pso = init(config, obj_f).unwrap();
    let term_condition = match terminate_f {
        Some(terminate_f) => terminate_f,
        None => |_| false,
    };
    pso.run(term_condition);
    Ok(pso)
}

/// Initializes and returns a PSO instance without running the optimization process
///
/// Useful for initializing an instance for running at a later time
pub fn init(
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
