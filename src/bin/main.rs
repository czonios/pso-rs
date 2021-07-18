use pso_rs::*;

fn main() {
    let config = Config {
        dimensions: vec![8],
        population_size: 100,
        bounds: vec![(-10.0, 10.0); 8],
        t_max: 1e7 as usize,
        ..Config::default()
    };
    use std::time::Instant;
    let before = Instant::now();
    let pso = pso_rs::run(config, sum_squares, None).unwrap();
    println!("Elapsed time: {:.2?}", before.elapsed());
    let model = pso.model;
    println!("Best f: {:#?} ", model.get_f_best());
    println!("Best x: {:#?} ", model.get_x_best());
}

fn sum_squares(p: &Particle, _flat_dim: usize, dimensions: &Vec<usize>) -> f64 {
    (0..dimensions[0]).map(|i| i as f64 * p[i].powf(2.0)).sum()
}
