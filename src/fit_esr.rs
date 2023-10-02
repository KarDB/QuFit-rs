use argmin::core::{CostFunction, Error, Executor, Gradient};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::BFGS;
use ndarray::{array, s, Array1, Array2, Array3};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

// Define the Lorentzian function
fn lorentzian(x: f64, params: &Array1<f64>) -> f64 {
    let a = params[0];
    let gamma = params[1];
    let x0 = params[2];

    a / std::f64::consts::PI * gamma / ((x - x0).powi(2) + gamma.powi(2))
}

// Define the cost function to be minimized
#[derive(Clone, Serialize, Deserialize)]
struct LorentzianCost {
    x_data: Array1<f64>,
    y_data: Array1<f64>,
}

impl CostFunction for LorentzianCost {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output, Error> {
        let residuals: Array1<f64> = &self.y_data - self.x_data.mapv(|x| lorentzian(x, params));
        Ok(residuals.dot(&residuals))
    }
}

impl Gradient for LorentzianCost {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(&self, params: &Self::Param) -> Result<Self::Gradient, Error> {
        let a = params[0];
        let gamma = params[1];
        let x0 = params[2];
        let pi = std::f64::consts::PI;
        let denom = (&self.x_data - x0).mapv(|x| x.powi(2)) + gamma.powi(2);

        let lor = 2.0 * (&self.y_data - &self.x_data.mapv(|x| lorentzian(x, params)));

        let df_da = 1.0 / pi * gamma / &denom * &lor;
        let df_dgamma = a / pi * ((&self.x_data - x0).mapv(|x| x.powi(2)) - gamma.powi(2))
            / denom.mapv(|x| x.powi(2))
            * &lor;
        let df_dx0 = -2.0 * a * (&self.x_data - x0) * gamma / pi / denom.mapv(|x| x.powi(2)) * &lor;

        Ok(array![df_da.sum(), df_dgamma.sum(), df_dx0.sum()])
    }
}

fn fit(x_data: Array1<f64>, y_data: Array1<f64>) -> Result<Array1<f64>, argmin::core::Error> {
    let cost = LorentzianCost { x_data, y_data };
    // Initial guess for the parameters [A, Γ, x₀]
    let init_params = array![0.5, 0.2, 0.6];
    let init_hessian: Array2<f64> = Array2::eye(3);
    // Set up the optimizer
    let linesearch = MoreThuenteLineSearch::new();
    let solver = BFGS::new(linesearch);
    let res = Executor::new(cost, solver)
        .configure(|state| {
            state
                .param(init_params)
                .inv_hessian(init_hessian)
                .max_iters(600)
        })
        .run()?;
    // Results
    let result = res.state.param;
    let opt_params = result.unwrap_or(array![0.1, 0.3, 0.5]);
    Ok(opt_params)
}

pub fn fit_image(data: Array3<f64>, y: Array1<f64>) -> Array3<f64> {
    let re: Array3<f64> = Array3::zeros((2, 2, 3));
    let re_mutex = Mutex::new(re);
    (0..2).into_par_iter().for_each(|i| {
        for j in 0..2 {
            let res = fit(data.slice(s![i, j, ..]).to_owned(), y.clone());
            let mut re = re_mutex.lock().unwrap();
            match res {
                Ok(result) => re.slice_mut(s![i, j, ..]).assign(&result),
                Err(err) => {
                    re.slice_mut(s![i, j, ..]).assign(&array![0.0, 0.0, 0.0]);
                    println!("The optmization failed {:?}", err);
                }
            }
        }
    });
    let res = re_mutex.into_inner().unwrap();
    res
}
