use levenberg_marquardt::{differentiate_numerically, LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::DVector;
use ndarray::{s, Array, Array3};
use ndarray_npy::read_npy;
use numpy::{IntoPyArray, PyArray3};
use pyo3::prelude::*;
mod fit_esr_nalgebra;
use fit_esr_nalgebra::LorenzianFit;

#[pymodule]
fn rust_lorentz(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit, m)?)?;
    Ok(())
}

#[pyfunction]
fn fit(py: Python, path: String) -> PyResult<PyObject> {
    let data: Array3<f64> = read_npy(path).unwrap();
    let x = Array::linspace(0.0, 1.0, 150);
    let data = DVector::from_vec(data.slice(s![1, 1, ..]).to_vec());
    let x = DVector::from_vec(x.to_vec());

    let mut problem = LorenzianFit {
        x_data: x,
        y_data: data,
        p: DVector::from_vec(vec![0.2, 0.2, 0.6]),
    };
    let jac = problem.jacobian().unwrap();
    println!("analytic jacobian {:?}", jac);
    let jac_num = differentiate_numerically(&mut problem).unwrap();
    println!("numerical jacobian {:?}", jac_num);
    // let res = fit_esr::fit_image(data, y);
    let (result, report) = LevenbergMarquardt::new().minimize(problem);
    println!("{:?}", result.p);
    dbg!(report.termination.was_successful());
    dbg!(report.objective_function.abs());
    dbg!(report.termination);
    dbg!(report.number_of_evaluations);
    let res: Array3<f64> = Array::zeros((3, 3, 3));
    let pyarray: &PyArray3<f64> = res.into_pyarray(py);
    Ok(pyarray.to_object(py))
}
