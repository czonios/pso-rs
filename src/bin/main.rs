use pso_rs::model::*;
use std::process;

fn main() {
    // let dimensions = vec![20, 3];
    // let population_size = 10;
    // let neighborhood_type = NeighborhoodType::Lbest;
    // let rho = 2;
    // let alpha = 0.08;
    // let lr = 0.5;
    // let c1 = 250.0;
    // let c2 = 0.8;
    // let bounds = (-2.5, 2.5);

    // let config = Config {
    //     dimensions,
    //     population_size,
    //     neighborhood_type,
    //     rho,
    //     alpha,
    //     c1,
    //     c2,
    //     lr,
    //     bounds,
    //     ..Config::default()
    // };

    // match pso_rs::run(config, e_lj) {
    //     Ok(mut pso) => {
    //         use std::time::Instant;
    //         let before = Instant::now();

    //         pso.run(pso.model.config.dimensions[0] * 1e5 as usize, |f_best| {
    //             f_best - (-77.177043) < 1e-4
    //         });
    //         println!("Elapsed time: {:.2?}", before.elapsed());
    //         // pso.write_to_file("./").unwrap_or_else(|err| {
    //         //     eprintln!("Problem writing trajectories: {}.", err);
    //         //     process::exit(1);
    //         // });
    //         let mut model = pso.model;
    //         println!("Model: {:?} ", model.get_error());
    //     }
    //     Err(e) => {
    //         eprintln!("Could not construct PSO: {}", e);
    //         process::exit(1);
    //     }
    // }

    // define objective function (Rosenbrock)
    fn objective_function(p: &Particle, _flat_dim: usize, _dimensions: &Vec<usize>) -> f64 {
        // x = p[0], y = p[1]
        (1.0 - p[0]).powf(2.0) + 100.0 * ((p[1] - p[0]).powf(2.0)).powf(2.0)
    }

    // define a termination condition
    fn terminate(f_best: f64) -> bool {
        f_best - (0.0) < 1e-6
    }

    let config = Config {
        dimensions: vec![2],
        population_size: 1000,
        bounds: (-5.0, 5.0),
        ..Config::default()
    };

    // define maximum number of objective function computations
    let t_max = 10000000;

    match pso_rs::run(config, objective_function) {
        Ok(mut pso) => {
            pso.run(t_max, terminate);
            let mut model = pso.model;
            println!("Model: {:?} ", model.get_f_best());
        }
        Err(e) => {
            eprintln!("Could not construct PSO: {}", e);
        }
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
