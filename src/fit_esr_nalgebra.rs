use argmm::generic::simple_argmin;
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{DMatrix, DVector, Dyn, Owned};
use ndarray::{array, s, Array1, Array3};
use rayon::prelude::*;
use std::sync::Mutex;

fn lorentzian(x: f64, params: &DVector<f64>) -> f64 {
    let a = params[0];
    let gamma = params[1];
    let x0 = params[2];
    1.0 - (a / std::f64::consts::PI * gamma / ((x - x0).powi(2) + gamma.powi(2)))
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
        let dl_da = -1.0 * &self.x_data.map(|x| self.gradient_da(x));
        let dl_dgamma = -1.0 * &self.x_data.map(|x| self.gradient_dgamma(x));
        let dl_dx0 = -1.0 * &self.x_data.map(|x| self.gradient_dx0(x));
        let jacobian =
            DMatrix::from_columns(&[dl_da.to_owned(), dl_dgamma.to_owned(), dl_dx0.to_owned()]);
        Some(jacobian)
    }
}

impl LorenzianFit {
    fn gradient_da(&self, x: f64) -> f64 {
        let params = self.params();
        let _a = params[0];
        let gamma = params[1];
        let x0 = params[2];
        let pi = std::f64::consts::PI;
        let denom = (x - x0).powi(2) + gamma.powi(2);

        let df_da = 1.0 / pi * gamma / denom;
        -1.0 * df_da
    }

    fn gradient_dgamma(&self, x: f64) -> f64 {
        let params = self.params();
        let a = params[0];
        let gamma = params[1];
        let x0 = params[2];
        let pi = std::f64::consts::PI;
        let denom = (x - x0).powi(2) + gamma.powi(2);

        let df_dgamma = a / pi * ((x - x0).powi(2) - gamma.powi(2)) / denom.powi(2);
        -1.0 * df_dgamma
    }

    fn gradient_dx0(&self, x: f64) -> f64 {
        let params = self.params();
        let a = params[0];
        let gamma = params[1];
        let x0 = params[2];
        let pi = std::f64::consts::PI;
        let denom = (x - x0).powi(2) + gamma.powi(2);

        let df_dx0 = 2.0 * a / pi * ((x - x0) * gamma) / denom.powi(2);
        -1.0 * df_dx0
    }
}

fn fit(x_data: Array1<f64>, y_data: Array1<f64>) -> Option<Array1<f64>> {
    let x_min_guess = simple_argmin(&y_data.to_vec()) as f64 / y_data.len() as f64;
    let init_param = vec![0.02, 0.02, x_min_guess];
    let x = DVector::from_vec(x_data.to_vec());
    let data = DVector::from_vec(y_data.to_vec());

    let problem = LorenzianFit {
        x_data: x,
        y_data: data,
        p: DVector::from_vec(init_param),
    };

    let (result, report) = LevenbergMarquardt::new().minimize(problem);
    let opt_params = result.p;
    let opt_params: Array1<f64> =
        Array1::from_shape_vec(opt_params.nrows(), opt_params.data.into()).unwrap();
    if report.termination.was_successful() {
        return Some(opt_params);
    } else {
        return None;
    }
}

pub fn fit_image(x_axis: Array1<f64>, data: Array3<f64>) -> Array3<f64> {
    let (xdim, ydim, _) = data.dim();
    let re: Array3<f64> = Array3::zeros((xdim, ydim, 3));
    let re_mutex = Mutex::new(re);
    (0..xdim).into_par_iter().for_each(|i| {
        for j in 0..ydim {
            let res = fit(x_axis.clone(), data.slice(s![i, j, ..]).to_owned());
            let mut re = re_mutex.lock().unwrap();
            match res {
                Some(result) => re.slice_mut(s![i, j, ..]).assign(&result),
                None => {
                    re.slice_mut(s![i, j, ..]).assign(&array![0.0, 0.0, 0.0]);
                    println!("The optmization failed! Assigning default zero values!");
                }
            }
        }
    });
    let res = re_mutex.into_inner().unwrap();
    res
}
