use pso_rs::Config;
use std::process;

fn main() {
    let dimensions = vec![20, 3];
    let population_size = 10;
    let neighborhood_type = "lbest";
    let rho = 2;
    let alpha = 0.08;
    let lr = 0.5;
    let c1 = 250.0;
    let c2 = 0.8;

    let config = Config::new(
        dimensions,
        population_size,
        neighborhood_type,
        rho,
        alpha,
        c1,
        c2,
        lr,
    )
    .unwrap_or_else(|err| {
        eprintln!("Problem parsing arguments: {}.", err);
        process::exit(1);
    });

    match pso_rs::run(config) {
        Ok(mut pso) => {
            use std::time::Instant;
            let before = Instant::now();

            pso.run(pso.model.config.dimensions[0] * 1e5 as usize, |f_best| {
                f_best - (-77.177043) < 1e-4
            });
            println!("Elapsed time: {:.2?}", before.elapsed());
            // pso.write_to_file("./").unwrap_or_else(|err| {
            //     eprintln!("Problem writing trajectories: {}.", err);
            //     process::exit(1);
            // });
            let mut model = pso.model;
            // model.population[0][0] = -0.3616353090;
            // model.population[0][1] = 0.0439914505;
            // model.population[0][2] = 0.5828840628;
            // model.population[0][3] = 0.2505889242;
            // model.population[0][4] = 0.6193583398;
            // model.population[0][5] = -0.1614607010;
            // model.population[0][6] = -0.4082757926;
            // model.population[0][7] = -0.2212115329;
            // model.population[0][8] = -0.5067996704;
            // model.population[0][9] = 0.5193221773;
            // model.population[0][10] = -0.4421382574;
            // model.population[0][11] = 0.0853763087;
            // println!("Population: {:#?} ", model.population);
            println!("Model: {:?} ", model.get_error());
        }
        Err(e) => {
            eprintln!("Application error: {}", e);
            process::exit(1);
        }
    }
}
