use crate::load::DataContainer;
use ndarray::{Array4, Axis, Dim}; //Zip
use ndrustfft::{ndfft_r2c_par, Complex, R2cFftHandler};

// When benchmarkding this reference function with criterion,
// the below option was the fastest by large margin. Needs to
// be adapted for this application though.
// fn reference_full_zip(
//     arr1: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 3]>>,
//     arr2: &ArrayBase<ViewRepr<&f32>, Dim<[usize; 3]>>,
// ) -> Array3<f32> {
//     let result = Zip::from(arr1)
//         .and(arr2)
//         .par_map_collect(|&a1, &a2| (a1 - a2) / (a1 + a2));
//     return result;
// }

impl DataContainer {
    pub fn array_fft(&self) -> Array4<Complex<f64>> {
        let dims: Vec<usize> = self.data.shape().iter().cloned().collect(); // this might be really
                                                                            // unecessary
        match dims.len() {
            5 => {
                dbg!(&dims);
                let mut vhat =
                    Array4::<Complex<f64>>::zeros((dims[1], dims[2] / 2 + 1, dims[3], dims[4]));
                let mut vhat = vhat.view_mut().into_dyn();
                let mut handler = R2cFftHandler::<f64>::new(dims[2]);
                let data = self.data.index_axis(Axis(0), 0);
                ndfft_r2c_par(&data, &mut vhat, &mut handler, 1);
                vhat.to_owned()
                    .into_dimensionality::<Dim<[usize; 4]>>()
                    .expect("Could not transform fft output into correct size array")
            }
            4 => {
                let mut vhat =
                    Array4::<Complex<f64>>::zeros((dims[1], dims[2] / 2 + 1, 1, dims[3]));
                let mut vhat = vhat.view_mut().into_dyn();
                let mut handler = R2cFftHandler::<f64>::new(dims[2]);
                let data = self.data.index_axis(Axis(0), 0);
                ndfft_r2c_par(&data, &mut vhat, &mut handler, 1);
                vhat.to_owned()
                    .into_dimensionality::<Dim<[usize; 4]>>()
                    .expect("Could not transform fft output into correct size array")
            }
            _ => unimplemented!("Your input has an unsupported number of dimensions (4, 5)"),
        }
    }
}
