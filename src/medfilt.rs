use crate::load::DataContainer;
use ndarray::{array, s, Array, Array2, Array3, ArrayView2, Axis};
use rayon::prelude::*;

fn medfilt2d(data: &ArrayView2<f64>, kernel_size: usize) -> Array2<f64> {
    let xdim = data.shape()[0];
    let ydim = data.shape()[1];
    let mut filtered: Array2<f64> = Array2::<f64>::zeros((xdim, ydim));
    // let medindex = kernel_size.pow(2) / 2 as usize;
    let kernel_size = (kernel_size - 1) / 2 as usize;
    for i in 0..xdim {
        // let xstart = i - kernel_size;
        // let xstart_clamped = xstart.max(0);
        let xstart_clamped = if i >= kernel_size { i - kernel_size } else { 0 };
        let xstop = i + kernel_size + 1;
        let xstop_clamped = xstop.min(xdim);
        for j in 0..ydim {
            // let ystart = j - kernel_size;
            // let ystart_clamped = ystart.max(0);
            let ystart_clamped = if j >= kernel_size { j - kernel_size } else { 0 };
            let ystop = j + kernel_size + 1;
            let ystop_clamped = ystop.min(ydim);
            let packet = data.slice(s![
                xstart_clamped..xstop_clamped,
                ystart_clamped..ystop_clamped
            ]);
            let mut flattened: Vec<f64> = packet.iter().cloned().collect();
            flattened.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let medindex = flattened.len() / 2 as usize;
            filtered[[i, j]] = flattened[medindex];
        }
    }
    filtered
}

impl DataContainer {
    pub fn medfilt_array(&self, kernel_size: usize) -> Array3<f64> {
        let zdim = self.data.shape()[2];
        let xdim = self.data.shape()[3];
        let ydim = self.data.shape()[4];
        let mut filtered = Array::zeros((zdim, xdim, ydim));

        filtered
            .axis_iter_mut(Axis(0))
            .into_par_iter() // Convert to a parallel iterator
            .enumerate() // Enumerate to get indices
            .for_each(|(i, mut subframe)| {
                // Apply some function to each sub-array
                let result = medfilt2d(&self.data.slice(s![0, 0, i, .., ..]), kernel_size);
                subframe.assign(&result);
            });

        filtered
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter() {
        let mut input = Array2::<f64>::zeros((10, 10));
        let mut slice = input.slice_mut(s![3..7, 3..7]);
        slice.fill(1.);
        let output = array![
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 1., 0., 0., 0., 0.],
            [0., 0., 0., 1., 1., 1., 1., 0., 0., 0.],
            [0., 0., 0., 1., 1., 1., 1., 0., 0., 0.],
            [0., 0., 0., 0., 1., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        ];
        assert_eq!(medfilt2d(&input.view(), 3), output);
    }
}
