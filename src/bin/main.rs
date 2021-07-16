use pso_rs::Config;
use std::process;

fn main() {
    let dimensions = vec![4, 3];
    let population_size = 2;
    let neighborhood_type = "lbest";
    let rho = 2;
    let alpha = 0.01;
    let c1 = 0.01;
    let c2 = 0.99;

    let config = Config::new(
        dimensions,
        population_size,
        neighborhood_type,
        rho,
        alpha,
        c1,
        c2,
    )
    .unwrap_or_else(|err| {
        eprintln!("Problem parsing arguments: {}.", err);
        process::exit(1);
    });

    match pso_rs::run(config) {
        Ok(mut pso) => {
            pso.run(20 /* 20 * 100000 */, |f_best| false);
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
