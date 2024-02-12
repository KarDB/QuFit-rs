mod fft;
mod fit_esr_nalgebra;
mod fit_rabi_nalgebra;
mod load;
mod medfilt;
use load::DataContainer;

fn main() {
    let mut data = DataContainer::new("data_grid.npy".into());
    dbg!(data.data.shape());
    data.compress_data(8);
    data.reference_ratio().unwrap();
    dbg!(data.data.shape());
}
