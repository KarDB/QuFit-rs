use load::DataContainer;
use numpy::PyArrayDyn;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
mod fft;
mod fit_esr_nalgebra;
mod fit_rabi_nalgebra;
mod load;
mod medfilt;
use numpy::IntoPyArray;

#[pymodule]
fn qufit(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<DataContainer>()?;
    m.add_function(wrap_pyfunction!(medfilt_pyth, m)?)?;
    Ok(())
}

#[pyfunction]
fn medfilt_pyth<'py>(
    py: Python<'py>,
    input: &PyArrayDyn<f64>,
    kernel_size: usize,
) -> &'py PyArrayDyn<f64> {
    let input_array = unsafe {
        input
            .as_array()
            .into_dimensionality::<ndarray::Ix3>()
            .expect("Input needs to be a 3D array")
    };
    let result = medfilt::medfilt_rust(&input_array.view(), kernel_size);
    result.into_dyn().into_pyarray(py)
}
