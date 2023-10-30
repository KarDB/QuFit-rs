use ndarray::{Array, Array3, Array4, Array5, Axis, Dimension, IxDyn, RemoveAxis, Slice};
use ndarray_npy::read_npy;
use std::sync::{Arc, Mutex};

// define dimension to be dynamic.
pub fn load_data(path: String) -> Array<f32, IxDyn> {
    let data: Array<u32, IxDyn> = read_npy(path).unwrap();
    let data = data.mapv(|x| x as f32);
    data
}

pub fn reference_ratio<D: Dimension>(data: Array<f32, D>) -> Result<Array<f32, D>, &'static str> {
    if data.len_of(Axis(0)) != 2 {
        return Err("The first axis should have a size of 2 for division");
    }
    let a0 = data.slice_axis(Axis(0), Slice::new(0, Some(1), 1));
    let a1 = data.slice_axis(Axis(0), Slice::new(1, Some(2), 1));
    Ok((&a0 / &a1).to_owned())
}

pub fn reference_sum<D: Dimension>(data: Array<f32, D>) -> Result<Array<f32, D>, &'static str> {
    if data.len_of(Axis(0)) != 2 {
        return Err("The first axis should have a size of 2 for division");
    }
    let a0 = data.slice_axis(Axis(0), Slice::new(0, Some(1), 1));
    let a1 = data.slice_axis(Axis(0), Slice::new(1, Some(2), 1));
    let res = (&a0 - &a1) / (&a0 + &a1);
    Ok(res.to_owned())
}

#[derive(Debug)]
pub struct DataContainer {
    data: Array<f32, IxDyn>,
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
}
