mod fit_esr_nalgebra;
mod load;
use load::DataContainer;
use load::{load_data, reference_ratio, reference_sum};

fn main() {
    // let data: Array3<f64> = read_npy("lorenzians.npy").unwrap();
    // let y = Array::linspace(0.0, 1.0, 150);
    // let data = DVector::from_vec(data.slice(s![1, 1, ..]).to_vec());
    // let y = DVector::from_vec(y.to_vec());
    // // let _res = fit_esr::fit_image(data, y);
    // let problem = LorenzianFit {
    //     x_data: data,
    //     y_data: y,
    //     p: DVector::from_vec(vec![0.2, 0.2, 0.2]),
    // };
    // let data = load_data("dataloading.npy".into());
    // dbg!(&data);
    // let ref_data = reference_ratio(data.clone());
    // dbg!(ref_data.unwrap());
    // let ref_data_sum = reference_sum(data);
    // dbg!(ref_data_sum.unwrap());
    //
    let mut data = DataContainer::new("dataloading.npy".into());
    dbg!(&data);
    data.reference_ratio();
    dbg!(&data);
    let mut data = DataContainer::new("dataloading.npy".into());
    data.reference_sum();
    dbg!(&data);
}
