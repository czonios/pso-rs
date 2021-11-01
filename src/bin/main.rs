use pso_rs::*;

const N_DIMENSIONS:usize = 3;

fn main() {
    let config = Config {
        dimensions: vec![N_DIMENSIONS],
        population_size: 100,
        bounds: vec![(-10.0, 10.0); N_DIMENSIONS],
        t_max: 1e7 as usize,
        ..Config::default()
    };
    use std::time::Instant;
    let before = Instant::now();
    let pso = pso_rs::run(config, sum_squares, Some(|f_best| f_best < 1e-4)).unwrap();
    println!("Elapsed time: {:.2?}", before.elapsed());
    let model = pso.model;
    println!("Found minimum: {:#?} ", model.get_f_best());
    println!("Found minimizer: {:#?} ", model.get_x_best());
}

fn sum_squares(p: &Particle, _flat_dim: usize, dimensions: &Vec<usize>) -> f64 {
    (0..dimensions[0]).map(|i| i as f64 * p[i].powf(2.0)).sum()
}
