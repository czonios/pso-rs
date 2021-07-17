use rand::{thread_rng, Rng};
use rayon::prelude::*;
use std::fmt;
pub type Particle = Vec<f64>;
pub type Population = Vec<Particle>;

/// Model struct
///
/// It takes in a `Config` instance and `fn` pointer to an objective function and defines a `run` method for running Particle Swarm Optimization.
pub struct Model {
    pub config: Config,
    pub flat_dim: usize,
    pub population: Population,
    pub population_f_scores: Vec<f64>,
    pub x_best: Particle,
    pub f_best: f64,
    obj_f: fn(&Particle, usize, &Vec<usize>) -> f64,
}

impl Model {
    pub fn new(
        config: Config,
        obj_f: fn(p: &Particle, flat_dim: usize, dim: &Vec<usize>) -> f64,
    ) -> Model {
        // init population
        let mut rng = thread_rng();
        let mut flat_dim = 1;
        for d in config.dimensions.clone() {
            flat_dim *= d;
        }
        let mut population: Population = vec![];
        for _ in 0..config.population_size {
            let mut particle: Particle = vec![];
            for _ in 0..flat_dim {
                particle.push(rng.gen_range(config.bounds.0..config.bounds.1));
            }
            population.push(particle);
        }
        let population_f_scores = vec![0.0; config.population_size];
        let x_best = population[0].clone();
        let f_best = population_f_scores[0].clone();
        let mut model = Model {
            config,
            flat_dim,
            population,
            population_f_scores,
            x_best,
            f_best,
            obj_f: obj_f,
        };
        model.get_f_values();
        model
    }

    /// Computes the value of the objective function for each particle and updates best found
    ///
    /// Returns the objective function values for all particles
    ///
    /// Uses the rayon crate for parallel computation
    pub fn get_f_values(&mut self) -> Vec<f64> {
        // find the objective function value for each member of the population
        let iter = self.population.par_iter();
        self.population_f_scores = iter
            .map(|particle| {
                (self.obj_f)(particle, self.flat_dim, &self.config.dimensions)
                // self.population_f_scores[i] = f_score;
            })
            .collect();
        // update best
        let mut f_best = self.population_f_scores[0];
        let mut x_best_index = 0;
        for (index, &score) in self.population_f_scores.iter().enumerate() {
            if score < f_best {
                f_best = score;
                x_best_index = index;
            }
        }
        self.f_best = f_best;
        self.x_best = self.population[x_best_index].clone();
        self.population_f_scores.to_owned()
    }

    /// Returns the best found objective function value
    pub fn get_f_best(&self) -> f64 {
        self.f_best
    }

    /// Returns the best found minimizer
    pub fn get_x_best(&self) -> Particle {
        self.x_best.clone()
    }
}

/// Configuration struct
///
/// Used to define model parameters
#[derive(Debug)]
pub struct Config {
    pub dimensions: Vec<usize>,
    pub population_size: usize,
    pub neighborhood_type: NeighborhoodType,
    pub rho: usize,
    pub alpha: f64,
    pub c1: f64,
    pub c2: f64,
    pub lr: f64,
    pub bounds: (f64, f64),
}

impl Config {
    pub fn new() -> Config {
        Self::default()
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dimensions: vec![2],
            population_size: 1000,
            neighborhood_type: NeighborhoodType::Lbest,
            rho: 2,
            alpha: 0.1,
            lr: 0.5,
            c1: 2.05,
            c2: 2.05,
            bounds: (-1.0, 1.0),
        }
    }
}

#[derive(Debug)]
pub enum NeighborhoodType {
    Lbest,
    Gbest,
}

impl fmt::Display for NeighborhoodType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NeighborhoodType::Lbest => write!(f, "Local neighborhood (lbest)"),
            NeighborhoodType::Gbest => write!(f, "Global neighborhood (gbest)"),
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn it_computes_correct_minimum() {
//         let dimensions = vec![4, 3];
//         let population_size = 1;
//         let neighborhood_type = NeighborhoodType::Lbest;
//         let rho = 1;
//         let alpha = 0.01;
//         let lr = 0.5;
//         let c1 = 0.01;
//         let c2 = 0.99;
//         let config = Config {
//             dimensions,
//             population_size,
//             neighborhood_type,
//             rho,
//             alpha,
//             c1,
//             c2,
//             lr,
//         };
//         let mut model = Model::new(config);

//         model.population[0][0] = -0.3616353090;
//         model.population[0][1] = 0.0439914505;
//         model.population[0][2] = 0.5828840628;
//         model.population[0][3] = 0.2505889242;
//         model.population[0][4] = 0.6193583398;
//         model.population[0][5] = -0.1614607010;
//         model.population[0][6] = -0.4082757926;
//         model.population[0][7] = -0.2212115329;
//         model.population[0][8] = -0.5067996704;
//         model.population[0][9] = 0.5193221773;
//         model.population[0][10] = -0.4421382574;
//         model.population[0][11] = 0.0853763087;

//         assert!(model.get_error() < -5.9999999);
//     }
// }
