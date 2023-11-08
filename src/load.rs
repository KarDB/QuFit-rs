use ndarray::{s, Array, ArrayD, Axis, Dimension, IxDyn, Slice};
use ndarray_npy::read_npy;
use std::cmp::min;

// define dimension to be dynamic.
pub fn load_data(path: String) -> Array<f32, IxDyn> {
    let data: Array<u32, IxDyn> = read_npy(path).unwrap();
    let data = data.mapv(|x| x as f32);
    data
}

#[derive(Debug)]
pub struct DataContainer {
    pub data: Array<f32, IxDyn>,
}

impl DataContainer {
    pub fn new(path: String) -> Self {
        DataContainer {
            data: load_data(path),
        }
    }

    pub fn reference_ratio(&mut self) -> Result<(), &'static str> {
        if self.data.len_of(Axis(0)) != 2 {
            return Err("The first axis should have a size of 2 for division");
        }
        let a0 = self.data.slice_axis(Axis(0), Slice::new(0, Some(1), 1));
        let a1 = self.data.slice_axis(Axis(0), Slice::new(1, Some(2), 1));
        self.data = (&a0 / &a1).to_owned();
        Ok(())
    }

    pub fn reference_sum(&mut self) -> Result<(), &'static str> {
        if self.data.len_of(Axis(0)) != 2 {
            return Err("The first axis should have a size of 2 for division");
        }
        let a0 = self.data.slice_axis(Axis(0), Slice::new(0, Some(1), 1));
        let a1 = self.data.slice_axis(Axis(0), Slice::new(1, Some(2), 1));
        self.data = ((&a0 - &a1) / (&a0 + &a1)).to_owned();
        Ok(())
    }

    pub fn get_new_shape(&self, stepsize: usize) -> Vec<usize> {
        let shape = self.data.shape().to_vec();
        let compressed_shape: Vec<_> = shape
            .iter()
            .skip(3)
            .map(|&s| (s + stepsize - 1) / stepsize)
            .collect();
        let new_shape: Vec<_> = shape
            .iter()
            .take(3)
            .cloned()
            .chain(compressed_shape)
            .collect();
        new_shape
    }

    pub fn blockwise_mean_2d(&self, stepsize: usize) -> ArrayD<f32> {
        let shape = self.data.shape();
        let max_index_3 = shape[3];
        let max_index_4 = shape[4];
        let new_shape = self.get_new_shape(stepsize);
        let mut result: ArrayD<f32> = Array::zeros(new_shape.as_slice());
        for new_idx in result.clone().indexed_iter() {
            let idx = new_idx.0.slice();
            let idx_vec = idx.to_vec();
            let slicer = s![
                idx_vec[0],
                idx_vec[1],
                idx_vec[2],
                stepsize * idx_vec[3]..min(stepsize * (idx_vec[3] + 1), max_index_3),
                stepsize * idx_vec[4]..min(stepsize * (idx_vec[4] + 1), max_index_4)
            ];
            let update_value = self.data.slice(slicer);
            result[idx] = update_value.mean().unwrap();
        }
        result
    }

    pub fn blockwise_mean_1d(&self, stepsize: usize) -> ArrayD<f32> {
        let shape = self.data.shape();
        let max_index_3 = shape[3];
        let new_shape = self.get_new_shape(stepsize);
        let mut result: ArrayD<f32> = Array::zeros(new_shape.as_slice());
        for new_idx in result.clone().indexed_iter() {
            let idx = new_idx.0.slice();
            let idx_vec = idx.to_vec();
            let slicer = s![
                idx_vec[0],
                idx_vec[1],
                idx_vec[2],
                stepsize * idx_vec[3]..min(stepsize * (idx_vec[3] + 1), max_index_3)
            ];
            let update_value = self.data.slice(slicer);
            result[idx] = update_value.mean().unwrap();
        }
        result
    }

    pub fn compress_data(&self, stepsize: usize) -> ArrayD<f32> {
        match self.data.ndim() {
            4 => {
                if self.data.shape().last().unwrap() == &1 {
                    self.data.clone()
                } else {
                    self.blockwise_mean_1d(stepsize)
                }
            }
            5 => self.blockwise_mean_2d(stepsize),
            _ => {
                panic!("The array has an unsupported number of dimensions. There is no method defined for this case. {:?}", self.data.shape())
            }
        }
    }
}
