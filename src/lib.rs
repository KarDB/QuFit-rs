use ndarray::{s, Array, Array3, Array4, Array5, Dimension, IxDyn};
use ndarray_npy::read_npy;
use numpy::{IntoPyArray, PyArray3};
use pyo3::prelude::*;
mod fit_esr_nalgebra;
mod fit_rabi_nalgebra;
//mod fit_t1_nalgebra;

#[pymodule]
fn rust_lorentz(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit, m)?)?;
    m.add_function(wrap_pyfunction!(fit_rabi, m)?)?;
    Ok(())
}

#[pyfunction]
fn fit(py: Python, path: String) -> PyResult<PyObject> {
    let data: Array3<f64> = read_npy(path).unwrap();
    let (_, _, zdim) = data.dim();
    let x_axis = Array::linspace(0.0, 1.0, zdim);
    let res = fit_esr_nalgebra::fit_image(x_axis, data);
    let pyarray: &PyArray3<f64> = res.into_pyarray(py);
    Ok(pyarray.to_object(py))
}

#[pyfunction]
fn fit_rabi(py: Python, path: String) -> PyResult<PyObject> {
    let data: Array3<f64> = read_npy(path).unwrap();
    let (_, _, zdim) = data.dim();
    let x_axis = Array::linspace(0.0, 1.0, zdim);
    let res = fit_rabi_nalgebra::fit_image(x_axis, data);
    let pyarray: &PyArray3<f64> = res.into_pyarray(py);
    Ok(pyarray.to_object(py))
}

// Define dimension to be dynamic.
fn load_data(path: String) -> Array<f32, IxDyn> {
    let data: Array<u32, IxDyn> = read_npy(path).unwrap();
    let data = data.mapv(|x| x as f32);
    data
}

fn reference_ratio<D: Dimension>(data: Array<f32, D>) -> () {}
