//! Module containing tensor with memory owned by Rust

use std::{ffi, fmt::Debug, ops::Deref};

use ndarray::Array;
use tracing::error;

use onnxruntime_sys as sys;

use crate::{
    error::{assert_not_null_pointer, call_ort, status_to_result},
    g_ort,
    memory::MemoryInfo,
    session::{AnyArray, NdArray},
    tensor::ndarray_tensor::NdArrayTensor,
    OrtError, Result, TensorElementDataType, TypeToTensorElementDataType,
};

/// Owned tensor, backed by an [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html)
///
/// This tensor bounds the ONNX Runtime to `ndarray`; it is used to copy an
/// [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html) to the runtime's memory.
///
/// **NOTE**: The type is not meant to be used directly, use an [`ndarray::Array`](https://docs.rs/ndarray/latest/ndarray/type.Array.html)
/// instead.
pub struct OrtTensor<'t, T, D>
where
    T: TypeToTensorElementDataType + Debug + Clone,
    D: ndarray::Dimension,
{
    #[allow(dead_code)]
    pub(crate) c_ptr: OrtValuePtr,
    array: Array<T, D>,

    #[allow(dead_code)]
    memory_info: &'t MemoryInfo,
}

/// To manage resource OrtValue pointer
#[derive(Debug)]
pub struct OrtValuePtr {
    pub(crate) c_ptr: *mut sys::OrtValue,
}

impl<'t, T, D> OrtTensor<'t, T, D>
where
    T: TypeToTensorElementDataType + Debug + Clone,
    D: ndarray::Dimension,
{
    #[allow(dead_code)]
    pub(crate) fn from_array<'m>(
        memory_info: &'m MemoryInfo,
        allocator_ptr: *mut sys::OrtAllocator,
        array: Array<T, D>,
    ) -> Result<OrtTensor<'t, T, D>>
    where
        'm: 't, // 'm outlives 't
    {
        let mut nd_array = NdArray::new(array);
        Ok(OrtTensor {
            c_ptr: OrtValuePtr::from_array(memory_info, allocator_ptr, &mut nd_array)?,
            array: nd_array.into(),
            memory_info,
        })
    }
}

impl<'t, T, D> Deref for OrtTensor<'t, T, D>
where
    T: TypeToTensorElementDataType + Debug + Clone,
    D: ndarray::Dimension,
{
    type Target = Array<T, D>;

    fn deref(&self) -> &Self::Target {
        &self.array
    }
}

impl OrtValuePtr {
    pub(crate) fn from_array(
        memory_info: &MemoryInfo,
        allocator_ptr: *mut sys::OrtAllocator,
        array: &mut dyn AnyArray,
    ) -> Result<Self> {
        let data_type = array.data_type();
        let shape: Vec<i64> = array.shape().iter().map(|d: &usize| *d as i64).collect();
        match data_type {
            TensorElementDataType::Float
            | TensorElementDataType::Uint8
            | TensorElementDataType::Int8
            | TensorElementDataType::Uint16
            | TensorElementDataType::Int16
            | TensorElementDataType::Int32
            | TensorElementDataType::Int64
            | TensorElementDataType::Double
            | TensorElementDataType::Uint32
            | TensorElementDataType::Uint64 => OrtValuePtr::from_number(
                data_type,
                memory_info,
                &shape,
                array.data_byte_len(),
                array.as_mut_void_ptr(),
            ),
            TensorElementDataType::String => OrtValuePtr::from_string(
                data_type,
                allocator_ptr,
                &shape,
                &array.to_null_terminated_strings()?,
            ),
        }
    }
    fn from_number(
        data_type: TensorElementDataType,
        memory_info: &MemoryInfo,
        shape: &[i64],
        array_len: usize,
        tensor_values_ptr: *mut std::ffi::c_void,
    ) -> Result<Self> {
        let mut tensor_ptr: *mut sys::OrtValue = std::ptr::null_mut();
        let tensor_ptr_ptr: *mut *mut sys::OrtValue = &mut tensor_ptr;
        let shape_ptr: *const i64 = shape.as_ptr();
        let shape_len = shape.len();
        assert_not_null_pointer(tensor_values_ptr, "TensorValues")?;

        unsafe {
            call_ort(|ort| {
                ort.CreateTensorWithDataAsOrtValue.unwrap()(
                    memory_info.ptr,
                    tensor_values_ptr,
                    array_len,
                    shape_ptr,
                    shape_len,
                    data_type.into(),
                    tensor_ptr_ptr,
                )
            })
        }
        .map_err(OrtError::CreateTensorWithData)?;
        assert_not_null_pointer(tensor_ptr, "Tensor")?;

        let mut is_tensor = 0;
        let status = unsafe { g_ort().IsTensor.unwrap()(tensor_ptr, &mut is_tensor) };
        status_to_result(status).map_err(OrtError::IsTensor)?;

        Ok(OrtValuePtr { c_ptr: tensor_ptr })
    }

    fn from_string(
        data_type: TensorElementDataType,
        allocator_ptr: *mut sys::OrtAllocator,
        shape: &[i64],
        null_terminated_copies: &[ffi::CString],
    ) -> Result<Self> {
        let mut tensor_ptr: *mut sys::OrtValue = std::ptr::null_mut();
        let tensor_ptr_ptr: *mut *mut sys::OrtValue = &mut tensor_ptr;
        let shape_ptr: *const i64 = shape.as_ptr();
        let shape_len = shape.len();
        unsafe {
            call_ort(|ort| {
                ort.CreateTensorAsOrtValue.unwrap()(
                    allocator_ptr,
                    shape_ptr,
                    shape_len,
                    data_type.into(),
                    tensor_ptr_ptr,
                )
            })
        }
        .map_err(OrtError::CreateTensor)?;
        let string_pointers = null_terminated_copies
            .iter()
            .map(|cstring| cstring.as_ptr())
            .collect::<Vec<_>>();

        unsafe {
            call_ort(|ort| {
                ort.FillStringTensor.unwrap()(
                    tensor_ptr,
                    string_pointers.as_ptr(),
                    string_pointers.len(),
                )
            })
        }
        .map_err(OrtError::FillStringTensor)?;
        Ok(OrtValuePtr { c_ptr: tensor_ptr })
    }
}

impl Drop for OrtValuePtr {
    #[tracing::instrument]
    fn drop(&mut self) {
        debug!("Dropping Tensor.");
        if self.c_ptr.is_null() {
            error!("Null pointer, not calling free.");
        } else {
            unsafe { g_ort().ReleaseValue.unwrap()(self.c_ptr) }
        }

        self.c_ptr = std::ptr::null_mut();
    }
}

impl<'t, T, D> OrtTensor<'t, T, D>
where
    T: TypeToTensorElementDataType + Debug + Clone,
    D: ndarray::Dimension,
{
    /// Apply a softmax on the specified axis
    pub fn softmax(&self, axis: ndarray::Axis) -> Array<T, D>
    where
        D: ndarray::RemoveAxis,
        T: ndarray::NdFloat + std::ops::SubAssign + std::ops::DivAssign,
    {
        self.array.softmax(axis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AllocatorType, MemType};
    use ndarray::{arr0, arr1, arr2, arr3};
    use std::ptr;
    use test_env_log::test;

    #[test]
    fn orttensor_from_array_0d_i32() {
        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
        let array = arr0::<i32>(123);
        let tensor = OrtTensor::from_array(&memory_info, ptr::null_mut(), array).unwrap();
        let expected_shape: &[usize] = &[];
        assert_eq!(tensor.shape(), expected_shape);
    }

    #[test]
    fn orttensor_from_array_1d_i32() {
        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
        let array = arr1(&[1_i32, 2, 3, 4, 5, 6]);
        let tensor = OrtTensor::from_array(&memory_info, ptr::null_mut(), array).unwrap();
        let expected_shape: &[usize] = &[6];
        assert_eq!(tensor.shape(), expected_shape);
    }

    #[test]
    fn orttensor_from_array_2d_i32() {
        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
        let array = arr2(&[[1_i32, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]);
        let tensor = OrtTensor::from_array(&memory_info, ptr::null_mut(), array).unwrap();
        assert_eq!(tensor.shape(), &[2, 6]);
    }

    #[test]
    fn orttensor_from_array_3d_i32() {
        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
        let array = arr3(&[
            [[1_i32, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],
            [[13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]],
            [[25, 26, 27, 28, 29, 30], [31, 32, 33, 34, 35, 36]],
        ]);
        let tensor = OrtTensor::from_array(&memory_info, ptr::null_mut(), array).unwrap();
        assert_eq!(tensor.shape(), &[3, 2, 6]);
    }

    #[test]
    fn orttensor_from_array_1d_string() {
        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
        let array = arr1(&[
            String::from("foo"),
            String::from("bar"),
            String::from("baz"),
        ]);
        let tensor = OrtTensor::from_array(&memory_info, ort_default_allocator(), array).unwrap();
        assert_eq!(tensor.shape(), &[3]);
    }

    #[test]
    fn orttensor_from_array_3d_str() {
        let memory_info = MemoryInfo::new(AllocatorType::Arena, MemType::Default).unwrap();
        let array = arr3(&[
            [["1", "2", "3"], ["4", "5", "6"]],
            [["7", "8", "9"], ["10", "11", "12"]],
        ]);
        let tensor = OrtTensor::from_array(&memory_info, ort_default_allocator(), array).unwrap();
        assert_eq!(tensor.shape(), &[2, 2, 3]);
    }

    fn ort_default_allocator() -> *mut sys::OrtAllocator {
        let mut allocator_ptr: *mut sys::OrtAllocator = std::ptr::null_mut();
        unsafe {
            // this default non-arena allocator doesn't need to be deallocated
            call_ort(|ort| ort.GetAllocatorWithDefaultOptions.unwrap()(&mut allocator_ptr))
        }
        .unwrap();
        allocator_ptr
    }
}
