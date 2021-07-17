use pso_rs::model::*;
use std::process;

fn main() {
    let dimensions = vec![4, 3];
    let population_size = 10;
    let neighborhood_type = NeighborhoodType::Lbest;
    let rho = 2;
    let alpha = 0.08;
    let lr = 0.5;
    let c1 = 250.0;
    let c2 = 0.8;
    let bounds = (-2.5, 2.5);

    let config = Config {
        dimensions,
        population_size,
        neighborhood_type,
        rho,
        alpha,
        c1,
        c2,
        lr,
        bounds,
        ..Config::default()
    };

    match pso_rs::run(config, e_lj) {
        Ok(mut pso) => {
            use std::time::Instant;
            let before = Instant::now();

            pso.run(pso.model.config.dimensions[0] * 1e5 as usize, |f_best| {
                f_best - (-77.177043) < 1e-4
            });
            println!("Elapsed time: {:.2?}", before.elapsed());
            pso.write_f_to_file("./best_f_trajectory.txt")
                .unwrap_or_else(|err| {
                    eprintln!("Problem writing trajectories: {}.", err);
                    process::exit(1);
                });
            pso.write_x_to_file("./best_x_trajectory.txt")
                .unwrap_or_else(|err| {
                    eprintln!("Problem writing trajectories: {}.", err);
                    process::exit(1);
                });
            let model = pso.model;
            println!("Model: {:?} ", model.get_f_best());
            // println!(
            //     "Best found minimizer: {:#?} ",
            //     reshape(&model.get_x_best(), &model.config.dimensions)
            // );
        }
        Err(e) => {
            eprintln!("Could not construct PSO: {}", e);
            process::exit(1);
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

fn reshape(particle: &Particle, particle_dims: &Vec<usize>) -> Vec<Vec<f64>> {
    // reshape particle
    let mut reshaped_population = vec![];
    let mut i = 0;
    for _ in 0..particle_dims[0] {
        let mut reshaped_particle = vec![];
        for _ in 0..particle_dims[1] {
            reshaped_particle.push(particle[i]);
            i += 1;
        }
        reshaped_population.push(reshaped_particle);
    }
    reshaped_population
}
