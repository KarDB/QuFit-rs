use argmm::generic::simple_argmin;
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{DMatrix, DVector, Dyn, Owned};
use ndarray::{array, s, Array1, Array3};
use rayon::prelude::*;
use std::sync::Mutex;

fn stretched_exponantial(t: f64, params: &DVector<f64>) -> f64 {
    let o = params[0];
    let a = params[1];
    let t_1 = params[2];
    let n = params[3];
    let exp_arg = -1.0 * (t / t_1).powf(n);
    o + a * exp_arg.exp()
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
        let residuals: DVector<f64> = &self.y_data
            - self
                .x_data
                .map(|x| stretched_exponantial(x, &self.params()));
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
        1.0
    }

    fn gradient_da(&self, t: f64) -> f64 {
        let params = self.params();
        let t_1 = params[2];
        let n = params[3];

        let exp_arg = -1.0 * (t / t_1).powf(n);
        let df_da = exp_arg.exp();
        df_da
    }

    fn gradient_dt_1(&self, t: f64) -> f64 {
        let params = self.params();
        let a = params[1];
        let t_1 = params[2];
        let n = params[3];

        let exp_arg = -1.0 * (t / t_1).powf(n);
        let df_dt_1 = a * exp_arg.exp() * n * (t / t_1).powf(n - 1.0) * (t / t_1.powi(2));
        df_dt_1
    }

    fn gradient_dn(&self, t: f64) -> f64 {
        let params = self.params();
        let a = params[1];
        let t_1 = params[2];
        let n = params[3];

        let exp_arg = -1.0 * (t / t_1).powf(n);
        let df_dn = a * exp_arg.exp() * n * (t / t_1).powf(n - 1.0) * (t / t_1.powi(2));
        df_dn
    }
}

fn fit(x_data: Array1<f64>, y_data: Array1<f64>) -> Option<Array1<f64>> {
    let y_data_tapered = (&y_data - y_data.mean().unwrap_or(1.0).clone()) * (-0.2 * &x_data + 1.0);
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

pub fn fit_image(x_axis: Array1<f64>, data: Array3<f64>) -> Array3<f64> {
    let (xdim, ydim, _) = data.dim();
    let re: Array3<f64> = Array3::zeros((xdim, ydim, 4));
    let re_mutex = Mutex::new(re);
    (0..xdim).into_par_iter().for_each(|i| {
        for j in 0..ydim {
            let res = fit(x_axis.clone(), data.slice(s![i, j, ..]).to_owned());
            let mut re = re_mutex.lock().unwrap();
            match res {
                Some(result) => re.slice_mut(s![i, j, ..]).assign(&result),
                None => {
                    re.slice_mut(s![i, j, ..])
                        .assign(&array![0.0, 0.0, 0.0, 0.0]);
                    println!("The optmization failed! Assigning default zero values!");
                }
            }
        }
    });
    let res = re_mutex.into_inner().unwrap();
    res
}
