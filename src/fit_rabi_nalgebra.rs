use crate::load::DataContainer;
use argmm::generic::simple_argmin;
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{DMatrix, DVector, Dyn, Owned};
use ndarray::{array, s, Array, Array1, Array3};
use rayon::prelude::*;
use std::sync::Mutex;

fn rabi(t: f64, params: &DVector<f64>) -> f64 {
    let o = params[0];
    let a = params[1];
    let tau = params[2];
    let phi = params[3];
    let cos_arg = std::f64::consts::PI / tau * t + phi;
    a * cos_arg.cos() + o
}

#[derive(Clone, Debug)]
pub struct RabiFit {
    pub x_data: DVector<f64>,
    pub y_data: DVector<f64>,
    pub p: DVector<f64>,
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for RabiFit {
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
        let residuals: DVector<f64> = &self.y_data - self.x_data.map(|x| rabi(x, &self.params()));
        Some(residuals)
    }

    fn jacobian(&self) -> Option<DMatrix<f64>> {
        let dl_do = &self.x_data.map(|x| self.gradient_do(x));
        let dl_da = &self.x_data.map(|x| self.gradient_da(x));
        let dl_dtau = &self.x_data.map(|x| self.gradient_dtau(x));
        let dl_dphi = &self.x_data.map(|x| self.gradient_dphi(x));
        let jacobian = DMatrix::from_columns(&[
            dl_do.to_owned(),
            dl_da.to_owned(),
            dl_dtau.to_owned(),
            dl_dphi.to_owned(),
        ]);
        Some(jacobian)
    }
}

impl RabiFit {
    fn gradient_do(&self, _t: f64) -> f64 {
        -1.0
    }

    fn gradient_da(&self, t: f64) -> f64 {
        let params = self.params();
        let tau = params[2];
        let phi = params[3];
        let pi = std::f64::consts::PI;

        let cos_arg = pi / tau * t + phi;
        let df_da = cos_arg.cos();
        -1.0 * df_da
    }

    fn gradient_dtau(&self, t: f64) -> f64 {
        let params = self.params();
        let a = params[1];
        let tau = params[2];
        let phi = params[3];
        let pi = std::f64::consts::PI;

        let sin_arg = pi / tau * t + phi;
        let df_dtau = pi / tau.powi(2) * t * a * sin_arg.sin();
        -1.0 * df_dtau
    }

    fn gradient_dphi(&self, t: f64) -> f64 {
        let params = self.params();
        let a = params[1];
        let tau = params[2];
        let phi = params[3];
        let pi = std::f64::consts::PI;

        let sin_arg = pi / tau * t + phi;
        let df_dphi = -1.0 * a * sin_arg.sin();
        -1.0 * df_dphi
    }
}

fn fit(x_data: Array1<f64>, y_data: Array1<f64>) -> Option<Array1<f64>> {
    let y_data_tapered = (&y_data - y_data.mean().unwrap_or(1.0).clone()) * (-0.4 * &x_data + 1.0);
    let tau_guess = simple_argmin(&y_data_tapered.to_vec()) as f64 / y_data.len() as f64;
    let offset = y_data.mean().unwrap_or(1.0);
    let init_param = vec![offset, 0.05, tau_guess, 0.0];
    let x = DVector::from_vec(x_data.to_vec());
    let data = DVector::from_vec(y_data.to_vec());

    let problem = RabiFit {
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

impl DataContainer {
    pub fn fit_rabi_image(&self) -> Array3<f64> {
        let zdim = self.data.shape()[2];
        let x_axis = Array::linspace(0.0, 1.0, zdim);
        let dims: Vec<usize> = self.data.shape().iter().cloned().skip(3).collect();
        match dims.len() {
            2 => {
                let xdim = dims[0];
                let ydim = dims[1];
                let re: Array3<f64> = Array3::zeros((xdim, ydim, 4));
                let re_mutex = Mutex::new(re);
                (0..xdim).into_par_iter().for_each(|i| {
                    for j in 0..ydim {
                        let res = fit(
                            x_axis.clone(),
                            self.data.slice(s![0, 0, .., i, j]).to_owned(),
                        );
                        let mut re = re_mutex.lock().unwrap();
                        match res {
                            Some(result) => re.slice_mut(s![i, j, ..]).assign(&result),
                            None => {
                                re.slice_mut(s![i, j, ..])
                                    .assign(&array![0.0, 0.0, 0.0, 0.0]);
                                println!("The optmization failed at {}, {}! Assigning default zero values!", i, j);
                            }
                        }
                    }
                });
                let res = re_mutex.into_inner().unwrap();
                res
            }
            1 => {
                let xdim = dims[0];
                let re: Array3<f64> = Array3::zeros((xdim, 1, 4));
                let re_mutex = Mutex::new(re);
                (0..xdim).into_par_iter().for_each(|i| {
                    let res = fit(x_axis.clone(), self.data.slice(s![0, 0, .., i]).to_owned());
                    let mut re = re_mutex.lock().unwrap();
                    match res {
                        Some(result) => re.slice_mut(s![i, 0, ..]).assign(&result),
                        None => {
                            re.slice_mut(s![i, 0, ..])
                                .assign(&array![0.0, 0.0, 0.0, 0.0]);
                            println!("The optmization failed! Assigning default zero values!");
                        }
                    }
                });
                let res = re_mutex.into_inner().unwrap();
                res
            }
            _ => panic!("For the size of the input array there are no known fitting methos"),
        }
    }
}
