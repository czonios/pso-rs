[![Crates.io](https://img.shields.io/crates/v/pso_rs?style=for-the-badge)](https://crates.io/crates/pso-rs)
[![docs.rs](https://img.shields.io/docsrs/pso-rs?style=for-the-badge)](https://docs.rs/pso-rs/latest/pso_rs/)
[![GitHub](https://img.shields.io/github/license/czonios/pso-rs?style=for-the-badge)](https://github.com/czonios/pso-rs/blob/master/LICENSE)

![Screenshot](https://raw.githubusercontent.com/czonios/pso-rs/master/screenshots/pbar.gif)

The [example](#example) below can get you started.
In order to use it in your own optimization problem, you will need to define an objective function as it is defined in the [`run`](https://docs.rs/pso-rs/latest/pso_rs/fn.run.html) function, and a [`Config`](https://docs.rs/pso-rs/latest/pso_rs/model/struct.Config.html) object. See the [documentation](https://docs.rs/pso-rs/latest/pso_rs/) for more information.

## Example

```rust
use pso_rs::*;

// define objective function (d-dimensional Rosenbrock)
fn objective_function(
    p: &Particle,
    _flat_dim: usize,
    dimensions: &Vec<usize>
) -> f64 {
    (0..dimensions[0] - 1).map(|i| {
        100.0 * ((p[i+1]-p[i]).powf(2.0)).powf(2.0)
            + (1.0-p[i]).powf(2.0)
    }).sum()
}

// define a termination condition (optional)
fn terminate(f_best: f64) -> bool {
    f_best - (0.0) < 1e-4
}

let config = Config {
    // dimension shape of each particle
    dimensions: vec![2],
    // problem bounds in each dimension
    bounds: vec![(-5.0, 10.0); 2],
    // maximum no. of objective function computations
    t_max: 10000,
    // leave the rest of the params as default
    ..Config::default()
};

let pso = pso_rs::run(
    config,
    objective_function,
    Some(terminate)
).unwrap();

let model = pso.model;
println!("Model: {:?} ", model.get_f_best());
```
