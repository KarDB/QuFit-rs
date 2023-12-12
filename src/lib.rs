use load::DataContainer;
use pyo3::prelude::*;
mod fft;
mod fit_esr_nalgebra;
mod fit_rabi_nalgebra;
mod load;

#[pymodule]
fn qufit(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<DataContainer>()?;
    Ok(())
}
