use crate::model::*;
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;

pub struct PSO {
    phi: f64,
    chi: f64,
    v_max: f64,
    pub model: Model,
    neighborhoods: Vec<Vec<usize>>,
    mut_population: Population,
    velocities: Population,
    neigh_population: Population,
    best_f_values: Vec<f64>,
}

impl PSO {
    pub fn new(mut model: Model) -> PSO {
        let phi = model.config.c1 + model.config.c2;
        let tmp = 2.0 - phi - (phi.powf(2.0) - (4.0 * phi));
        let tmp = tmp.sqrt();
        let chi = 2.0 / (2.0 - phi - tmp).abs();
        let v_max = model.config.alpha * 5.0;
        let mut neighborhoods = vec![];
        match model.config.neighborhood_type {
            NeighborhoodType::Lbest => {
                neighborhoods = vec![];
                for i in 0..model.config.population_size {
                    let mut neighbor = vec![];
                    let first_neighbor = i as i32 - model.config.rho as i32;
                    let last_neighbor = i as i32 + model.config.rho as i32;

                    for neighbor_i in first_neighbor..last_neighbor {
                        neighbor.push(if neighbor_i < 0 {
                            (model.config.population_size as i32 - neighbor_i) as usize
                        } else {
                            neighbor_i as usize
                        });
                    }
                    neighborhoods.push(neighbor)
                }
            }
            NeighborhoodType::Gbest => {
                neighborhoods = vec![];
                for _ in 0..model.config.population_size {
                    let mut tmp = vec![];
                    for j in 0..model.config.population_size {
                        tmp.push(j);
                    }
                    neighborhoods.push(tmp);
                }
            }
        }

        // initialize
        let mut_population = model.population.clone();
        let mut velocities = vec![];
        for _ in 0..model.config.population_size {
            let mut tmp = vec![];
            for _ in 0..model.flat_dim {
                tmp.push(model.rng.gen_range(-1.0..1.0));
            }
            velocities.push(tmp);
        }

        let best_f_values = model.population_f_scores.clone();
        let neigh_population = model.population.clone();

        PSO {
            phi,
            chi,
            v_max,
            model,
            neighborhoods,
            mut_population,
            velocities,
            best_f_values,
            neigh_population,
        }
    }

    fn argsort(v: &Vec<f64>) -> Vec<usize> {
        let mut idx = (0..v.len()).collect::<Vec<_>>();
        idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).expect("NaN"));
        idx
    }

    fn local_best(&self, i: usize) -> usize {
        let best = PSO::argsort(&self.best_f_values);
        for b in best {
            if self.neighborhoods[i].iter().any(|&n| n == b) {
                return b;
            }
        }
        0
    }

    fn update_velocity_and_pos(&mut self) {
        for i in 0..self.model.config.population_size {
            let lbest = &self.neigh_population[self.local_best(i)];
            for j in 0..self.model.flat_dim {
                let r1 = self.model.rng.gen_range(-1.0..1.0);
                let r2 = self.model.rng.gen_range(-1.0..1.0);
                let cog = self.model.config.c1
                    * r1
                    * (self.neigh_population[i][j] - self.mut_population[i][j]);

                let soc = self.model.config.c2 * r2 * (lbest[j] - self.mut_population[i][j]);

                let v = self.chi * (self.velocities[i][j] + cog + soc);
                // check bounds
                self.velocities[i][j] = if v.abs() > self.v_max {
                    v
                } else {
                    v.signum() * self.v_max
                };

                let x = self.mut_population[i][j] + self.velocities[i][j];
                // check bounds
                if x > 2.5 {
                    self.mut_population[i][j] = 2.5; // TODO dynamic bounds from config
                } else if x < -2.5 {
                    self.mut_population[i][j] = -2.5; // TODO dynamic bounds from config
                } else {
                    self.mut_population[i][j] = x;
                }
            }
        }
    }

    fn update_best_positions(&mut self) {
        for i in 0..self.best_f_values.len() {
            let new = self.model.population_f_scores[i];
            let old = self.best_f_values[i];

            if new < old {
                self.best_f_values[i] = new;
                self.neigh_population[i] = self.mut_population[i].clone();
                // check if global best found
                if new < self.model.f_best {
                    self.model.f_best = new;
                    self.model.x_best_index = i;
                }
            }
        }
    }

    pub fn run(&mut self, t_max: usize, terminate: fn(f64) -> bool) -> usize {
        let bar = ProgressBar::new(t_max as u64);
        bar.set_style(
            ProgressStyle::default_bar()
                .template("{msg} [{elapsed}] {bar:20.cyan/blue} {pos:>7}/{len:7} ETA: {eta}"),
        );
        let mut k = 0;
        let pop_size = self.model.config.population_size;
        while k < t_max && !terminate(self.model.f_best) {
            // Update velocity and positions
            self.update_velocity_and_pos();

            // Evaluate & update best
            self.model.get_f_values();
            println!(
                "self.model.get_f_values(): {:#?} ",
                self.model.get_f_values()
            );
            self.update_best_positions();

            self.model.population = self.mut_population.clone();
            k += pop_size;
            bar.inc(pop_size as u64);
            bar.set_message(format!("{:.4}", self.model.f_best));
        }

        bar.finish();
        k
    }
}
