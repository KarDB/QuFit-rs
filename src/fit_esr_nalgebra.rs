use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{DMatrix, DVector, Dyn, Owned};
// use rayon::prelude::*;
// use std::sync::Mutex;

fn lorentzian(x: f64, params: &DVector<f64>) -> f64 {
    let a = params[0];
    let gamma = params[1];
    let x0 = params[2];
    a / std::f64::consts::PI * gamma / ((x - x0).powi(2) + gamma.powi(2))
}

#[derive(Clone, Debug)]
pub struct LorenzianFit {
    pub x_data: DVector<f64>,
    pub y_data: DVector<f64>,
    pub p: DVector<f64>,
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for LorenzianFit {
    type ParameterStorage = Owned<f64, Dyn>;
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Dyn>;

    fn set_params(&mut self, p: &DVector<f64>) {
        self.p.copy_from(p)
    }

    fn params(&self) -> DVector<f64> {
        self.p.clone()
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        let residuals: DVector<f64> =
            &self.y_data - self.x_data.map(|x| lorentzian(x, &self.params()));
        Some(residuals)
    }

    fn jacobian(&self) -> Option<DMatrix<f64>> {
        let dl_da = self.gradient_da();
        let dl_dgamma = self.gradient_dgamma();
        let dl_dx0 = self.gradient_dx0();
        let jacobian = DMatrix::from_columns(&[dl_da, dl_dgamma, dl_dx0]);
        Some(jacobian)
    }
}

impl LorenzianFit {
    fn gradient_da(&self) -> DVector<f64> {
        let params = self.params();
        let _a = params[0];
        let gamma = params[1];
        let x0 = params[2];
        let pi = std::f64::consts::PI;
        let denom = (&self.x_data.map(|x| x - x0))
            .map(|x| x.powi(2))
            .map(|x| x + gamma.powi(2));
        let lor = 2.0 * (&self.y_data - &self.x_data.map(|x| lorentzian(x, &params)));
        let denom_lor = &lor.zip_map(&denom, |l, d| l / d);
        let df_da = denom_lor.map(|dl| 1.0 / pi * gamma * dl);
        df_da
    }

    fn gradient_dgamma(&self) -> DVector<f64> {
        let params = self.params();
        let a = params[0];
        let gamma = params[1];
        let x0 = params[2];
        let pi = std::f64::consts::PI;
        let denom = (&self.x_data.map(|x| x - x0))
            .map(|x| x.powi(2))
            .map(|x| x + gamma.powi(2));
        let lor = 2.0 * (&self.y_data - &self.x_data.map(|x| lorentzian(x, &params)));

        let denomsquare = denom.map(|x| x.powi(2));
        let denomsquare_lor = &lor.zip_map(&denomsquare, |l, d| l / d);
        let derivative_numerator = (&self.x_data.map(|x| x - x0))
            .map(|x| x.powi(2))
            .map(|x| x - gamma.powi(2));
        let df_dgamma = denomsquare_lor.zip_map(&derivative_numerator, |dl, num| a / pi * num * dl);
        df_dgamma
    }

    fn gradient_dx0(&self) -> DVector<f64> {
        let params = self.params();
        let a = params[0];
        let gamma = params[1];
        let x0 = params[2];
        let pi = std::f64::consts::PI;
        let denom = (&self.x_data.map(|x| x - x0))
            .map(|x| x.powi(2))
            .map(|x| x + gamma.powi(2));
        let lor = 2.0 * (&self.y_data - &self.x_data.map(|x| lorentzian(x, &params)));
        let denom_square = denom.map(|x| x.powi(2));
        let denomsquare_lor = lor.zip_map(&denom_square, |l, d| l / d);
        let x_diff = &self.x_data.map(|x| x - x0);
        let df_dx0 = denomsquare_lor.zip_map(&x_diff, |dsl, x| -2.0 * a * gamma / pi * dsl * x);
        df_dx0
    }
}

// fn fit(x_data: Array1<f64>, y_data: Array1<f64>) -> Result<Array1<f64>, argmin::core::Error> {
//     let cost = LorentzianCost { x_data, y_data };
//     // Initial guess for the parameters [A, Γ, x₀]
//     let init_params = array![0.5, 0.2, 0.6];
//     let init_hessian: Array2<f64> = Array2::eye(3);
//     // Set up the optimizer
//     let linesearch = MoreThuenteLineSearch::new();
//     let solver = BFGS::new(linesearch);
//     let res = Executor::new(cost, solver)
//         .configure(|state| {
//             state
//                 .param(init_params)
//                 .inv_hessian(init_hessian)
//                 .max_iters(600)
//         })
//         .run()?;
//     // Results
//     let result = res.state.param;
//     let opt_params = result.unwrap_or(array![0.1, 0.3, 0.5]);
//     Ok(opt_params)
// }
//
// pub fn fit_image(data: Array3<f64>, y: Array1<f64>) -> Array3<f64> {
//     let re: Array3<f64> = Array3::zeros((2, 2, 3));
//     let re_mutex = Mutex::new(re);
//     (0..2).into_par_iter().for_each(|i| {
//         for j in 0..2 {
//             let res = fit(data.slice(s![i, j, ..]).to_owned(), y.clone());
//             let mut re = re_mutex.lock().unwrap();
//             match res {
//                 Ok(result) => re.slice_mut(s![i, j, ..]).assign(&result),
//                 Err(err) => {
//                     re.slice_mut(s![i, j, ..]).assign(&array![0.0, 0.0, 0.0]);
//                     println!("The optmization failed {:?}", err);
//                 }
//             }
//         }
//     });
//     let res = re_mutex.into_inner().unwrap();
//     res
// }
