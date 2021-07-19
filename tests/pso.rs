use pso_rs::*;

#[test]
fn it_computes_correct_minimum_rosenbrock_2d() {
    fn rosenbrock(p: &Particle, _flat_dim: usize, dimensions: &Vec<usize>) -> f64 {
        (0..dimensions[0] - 1)
            .map(|i| 100.0 * ((p[i + 1] - p[i]).powf(2.0)).powf(2.0) + (1.0 - p[i]).powf(2.0))
            .sum()
    }

    let config = Config {
        t_max: 1,
        population_size: 1,
        progress_bar: false,
        ..Config::default()
    };
    let pso = pso_rs::run(config, rosenbrock, None).unwrap();

    let mut model = pso.model;

    model.population[0][0] = 2.0;
    model.population[0][1] = -2.0;
    model.get_f_values();

    assert_ne!(model.get_f_best(), 0.0);

    model.population[0][0] = 1.0;
    model.population[0][1] = 1.0;
    model.get_f_values();

    assert_eq!(model.get_f_best(), 0.0);
}

#[test]
fn it_computes_correct_minimum_rosenbrock_3d() {
    fn rosenbrock(p: &Particle, _flat_dim: usize, dimensions: &Vec<usize>) -> f64 {
        (0..dimensions[0] - 1)
            .map(|i| 100.0 * ((p[i + 1] - p[i]).powf(2.0)).powf(2.0) + (1.0 - p[i]).powf(2.0))
            .sum()
    }

    let config = Config {
        dimensions: vec![3],
        t_max: 1,
        bounds: vec![(-5.0, 10.0); 3],
        population_size: 1,
        progress_bar: false,
        ..Config::default()
    };
    let pso = pso_rs::run(config, rosenbrock, None).unwrap();

    let mut model = pso.model;

    model.population[0][0] = 2.0;
    model.population[0][1] = -2.0;
    model.population[0][2] = -2.0;
    model.get_f_values();

    assert_ne!(model.get_f_best(), 0.0);

    model.population[0][0] = 1.0;
    model.population[0][1] = 1.0;
    model.population[0][2] = 1.0;
    model.get_f_values();

    assert_eq!(model.get_f_best(), 0.0);
}

#[test]
fn it_computes_correct_minimum_e_lj() {
    /// Get Euclidian distance of two particles
    fn l2(x_i: Particle, x_j: Particle, particle_dim: usize) -> f64 {
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
    let config = Config {
        dimensions: vec![4, 3],
        bounds: vec![(-2.5, 2.5); 3],
        population_size: 1,
        progress_bar: false,
        ..Config::default()
    };

    let pso = pso_rs::run(config, e_lj, Some(|_| true)).unwrap();

    let mut model = pso.model;

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
    model.get_f_values();
    assert!(model.get_f_best() < -5.9999999);
}
