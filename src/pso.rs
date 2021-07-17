use crate::model::*;
use indicatif::{ProgressBar, ProgressStyle};
use rand::{thread_rng, Rng};

use std::error::Error;
use std::fs::File;
use std::io::Write;

pub struct PSO {
    chi: f64,
    v_max: f64,
    pub model: Model,
    neighborhoods: Vec<Vec<usize>>,
    mut_population: Population,
    velocities: Population,
    neigh_population: Population,
    best_f_values: Vec<f64>,
    best_f_trajectory: Vec<f64>,
    best_x_trajectory: Vec<Particle>,
}

impl PSO {
    /// Initialize Particle Swarm Optimization
    pub fn new(model: Model) -> PSO {
        let phi = model.config.c1 + model.config.c2;
        let phi_squared = phi.powf(2.0);
        let tmp = phi_squared - (4.0 * phi);
        let tmp = tmp.sqrt();
        let chi = 2.0 / (2.0 - phi - tmp).abs();
        let v_max = model.config.alpha * 5.0;
        let mut neighborhoods;
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
        let mut rng = thread_rng();
        let mut_population = model.population.clone();
        let mut velocities = vec![];
        for _ in 0..model.config.population_size {
            let mut tmp = vec![];
            for _ in 0..model.flat_dim {
                tmp.push(rng.gen_range(-1.0..1.0));
            }
            velocities.push(tmp);
        }

        let best_f_values = model.population_f_scores.clone();
        let neigh_population = model.population.clone();
        let best_f_trajectory = vec![model.f_best];
        let best_x_trajectory = vec![model.x_best.clone()];

        PSO {
            chi,
            v_max,
            model,
            neighborhoods,
            mut_population,
            velocities,
            best_f_values,
            neigh_population,
            best_f_trajectory,
            best_x_trajectory,
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
        let mut rng = thread_rng();

        for i in 0..self.model.config.population_size {
            let lbest = &self.neigh_population[self.local_best(i)];
            for j in 0..self.model.flat_dim {
                let r1 = rng.gen_range(-1.0..1.0);
                let r2 = rng.gen_range(-1.0..1.0);
                let cog = self.model.config.c1
                    * r1
                    * (self.neigh_population[i][j] - self.mut_population[i][j]);

                let soc = self.model.config.c2 * r2 * (lbest[j] - self.mut_population[i][j]);
                let v = self.chi * (self.velocities[i][j] + cog + soc);

                // check bounds
                self.velocities[i][j] = if v.abs() > self.v_max {
                    v.signum() * self.v_max
                } else {
                    v
                };

                let x = self.mut_population[i][j] + self.model.config.lr * self.velocities[i][j];
                // check bounds
                if x > self.model.config.bounds.1 {
                    self.mut_population[i][j] = self.model.config.bounds.1; // TODO dynamic bounds from config
                } else if x < self.model.config.bounds.0 {
                    self.mut_population[i][j] = self.model.config.bounds.0; // TODO dynamic bounds from config
                } else {
                    self.mut_population[i][j] = x;
                }
                if x.is_nan() {
                    panic!("A coefficient became NaN!");
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
                    self.model.x_best = self.model.population[i].clone();
                }
            }
        }
        self.best_f_trajectory.push(self.model.f_best);
        self.best_x_trajectory.push(self.model.x_best.clone());
    }

    /// Performs Particle Swarm Optimization
    ///
    /// # Panics
    ///
    /// Panics if any particle coefficient becomes NaN
    pub fn run(&mut self, terminate: fn(f64) -> bool) -> usize {
        let bar = ProgressBar::new(self.model.config.t_max as u64);
        bar.set_style(
            ProgressStyle::default_bar()
                .template("{msg} [{elapsed}] {bar:20.cyan/blue} {pos:>7}/{len:7} ETA: {eta}"),
        );
        let mut k = 0;
        let pop_size = self.model.config.population_size;
        loop {
            // Update velocity and positions
            self.update_velocity_and_pos();

            // Evaluate & update best
            self.model.get_f_values();
            self.update_best_positions();

            self.model.population = self.mut_population.clone();
            k += pop_size;
            bar.inc(pop_size as u64);
            bar.set_message(format!("{:.6}", self.model.f_best));
            if k > self.model.config.t_max || terminate(self.model.f_best) {
                break;
            }
        }

        bar.finish_and_clear();
        k
    }

    /// Writes the best found objective function value for all iterations separated by newline characters
    pub fn write_f_to_file(&self, filepath: &str) -> Result<(), Box<dyn Error>> {
        let best_f_str: Vec<String> = self
            .best_f_trajectory
            .iter()
            .map(|n| n.to_string())
            .collect();

        let mut file = File::create(filepath)?;
        writeln!(file, "{}", best_f_str.join("\n"))?;

        Ok(())
    }

    /// Writes the best found minimizer for all iterations
    ///
    /// Vector coefficients are comma-separated, and the best vector at each iteration is terminated with a newline character
    pub fn write_x_to_file(&self, filepath: &str) -> Result<(), Box<dyn Error>> {
        let best_x_str: Vec<String> = self
            .best_x_trajectory
            .iter()
            .map(|x| {
                x.iter()
                    .map(|coef: &f64| coef.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            })
            .collect();

        let mut file = File::create(filepath)?;
        writeln!(file, "{}", best_x_str.join("\n"))?;

        Ok(())
    }
}
