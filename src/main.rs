mod fit_esr_nalgebra;
use fit_esr_nalgebra::LorenzianFit;
use nalgebra::DVector;
use ndarray::{s, Array, Array3};
use ndarray_npy::read_npy;

fn main() {
    let data: Array3<f64> = read_npy("lorenzians.npy").unwrap();
    let y = Array::linspace(0.0, 1.0, 150);
    let data = DVector::from_vec(data.slice(s![1, 1, ..]).to_vec());
    let y = DVector::from_vec(y.to_vec());
    // let _res = fit_esr::fit_image(data, y);
    let problem = LorenzianFit {
        x_data: data,
        y_data: y,
        p: DVector::from_vec(vec![0.2, 0.2, 0.2]),
    };
}
