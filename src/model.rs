use rand::{thread_rng, Rng};
use rayon::prelude::*;
pub type Particle = Vec<f64>;
pub type Population = Vec<Particle>;

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
    pub fn new(config: Config) -> Model {
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
                particle.push(rng.gen_range(-2.5..2.5));
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
            obj_f: |p, dim, flat_dim| e_lj(p, dim, flat_dim),
        };
        model.get_f_values();
        model
    }

    /// Returns the value of the objective function for each cluster
    /// Notes:
    ///    - uses the whole population by default
    ///    - used by both GA and PSO
    ///
    /// Returns:
    ///    float[]: values of the objective function for the input clusters
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

    pub fn get_error(&mut self) -> f64 {
        self.get_f_values();
        self.f_best
    }
}

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
}

impl Config {
    pub fn new(
        dimensions: Vec<usize>,
        population_size: usize,
        neighborhood_type: &str,
        rho: usize,
        alpha: f64,
        c1: f64,
        c2: f64,
        lr: f64,
    ) -> Result<Config, &'static str> {
        let neighborhood_type = match neighborhood_type {
            arg => match &arg.to_lowercase()[..] {
                "lbest" => NeighborhoodType::Lbest,
                "gbest" => NeighborhoodType::Gbest,
                _ => return Err("Only `lbest` and `gbest` are valid neighborhood types"),
            },
        };
        // param: mut args: env::Args
        // if args.len() < 7 {
        //     return Err("Please specify all arguments");
        // }
        // args.next();
        // let dimensions = match args.next() {
        //     Some(arg) => match arg.parse() {
        //         Ok(arg) => arg,
        //         _ => return Err("Please specify dimensions of each particle"),
        //     },
        //     None => return Err("Please specify dimensions of each particle"),
        // };

        // let population_size = match args.next() {
        //     Some(arg) => match arg.parse() {
        //         Ok(arg) => arg,
        //         _ => return Err("Please specify population size"),
        //     },
        //     None => return Err("Please specify population size"),
        // };
        // let neighborhood_type = match args.next() {
        //     Some(arg) => match &arg.to_lowercase()[..] {
        //         "lbest" => NeighborhoodType::Lbest,
        //         "gbest" => NeighborhoodType::Gbest,
        //         _ => return Err("Only `lbest` and `gbest` are valid neighborhood types"),
        //     },
        //     None => return Err("Please specify neighborhood type"),
        // };
        // let rho = match args.next() {
        //     Some(arg) => match arg.parse() {
        //         Ok(arg) => arg,
        //         _ => return Err("Please specify rho (number of neighbors on each side)"),
        //     },
        //     None => return Err("Please specify rho (number of neighbors on each side)"),
        // };
        // let alpha = match args.next() {
        //     Some(arg) => match arg.parse() {
        //         Ok(arg) => arg,
        //         _ => return Err("Please specify alpha (max velocity parameter)"),
        //     },
        //     None => return Err("Please specify alpha (max velocity parameter)"),
        // };
        // let c1 = match args.next() {
        //     Some(arg) => match arg.parse() {
        //         Ok(arg) => arg,
        //         _ => return Err("Please specify C1"),
        //     },
        //     None => return Err("Please specify C1"),
        // };
        // let c2 = match args.next() {
        //     Some(arg) => match arg.parse() {
        //         Ok(arg) => arg,
        //         _ => return Err("Please specify C2"),
        //     },
        //     None => return Err("Please specify C2"),
        // };

        Ok(Config {
            dimensions,
            population_size,
            neighborhood_type,
            rho,
            alpha,
            c1,
            c2,
            lr,
        })
    }
}

/// Get Euclidian distance of two particles
fn l2(x_i: Particle, x_j: Particle, particle_dim: usize) -> f64 {
    // calculated as the square root of the sum of the squared vector values
    let mut sum: f64 = 0.0;
    for i in 0..particle_dim {
        sum += (x_i[i] - x_j[i]).powf(2.0);
    }
    sum.sqrt()
}

/// Get potential energy of two particles
fn v_ij(x_i: Particle, x_j: Particle, particle_dim: usize) -> f64 {
    let denom: f64 = 1.0 / l2(x_i, x_j, particle_dim);
    denom.powf(12.0) - denom.powf(6.0)
}

/// Get potential energy of a cluster of particles
fn e_lj(particle: &Particle, _flat_dim: usize, particle_dims: &Vec<usize>) -> f64 {
    // reshape particle

    let mut sum = 0.0;
    for i in 0..particle_dims[0] - 1 {
        for j in (i + 1)..particle_dims[0] {
            let true_i = i * particle_dims[1];
            let true_j = j * particle_dims[1];
            sum += v_ij(
                particle[true_i..true_i + particle_dims[1]].to_vec(),
                particle[true_j..true_j + particle_dims[1]].to_vec(),
                particle_dims[1],
            );
        }
    }
    4.0 * sum
}

#[derive(Debug)]
pub enum NeighborhoodType {
    Lbest,
    Gbest,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_computes_correct_minimum() {
        let dimensions = vec![4, 3];
        let population_size = 1;
        let neighborhood_type = NeighborhoodType::Lbest;
        let rho = 1;
        let alpha = 0.01;
        let lr = 0.5;
        let c1 = 0.01;
        let c2 = 0.99;
        let config = Config {
            dimensions,
            population_size,
            neighborhood_type,
            rho,
            alpha,
            c1,
            c2,
            lr,
        };
        let mut model = Model::new(config);

        model.population[0][0] = -0.3616353090;
        model.population[0][1] = 0.0439914505;
        model.population[0][2] = 0.5828840628;
        model.population[0][3] = 0.2505889242;
        model.population[0][4] = 0.6193583398;
        model.population[0][5] = -0.1614607010;
        model.population[0][6] = -0.4082757926;
        model.population[0][7] = -0.2212115329;
        model.population[0][8] = -0.5067996704;
        model.population[0][9] = 0.5193221773;
        model.population[0][10] = -0.4421382574;
        model.population[0][11] = 0.0853763087;

        assert!(model.get_error() < -5.9999999);
    }
}
