use std::ffi::{c_char, c_int, c_void};

use libloading::{Library, Symbol};

pub type CviRc = c_int;
pub type CviModelHandle = *mut c_void;

#[repr(C)]
#[derive(Debug)]
pub struct CviShape {
    pub dim: [i32; 8],
    pub dim_size: i32,
}

#[repr(C)]
pub struct CviTensor {
    pub name: [u8; 64],
    pub shape: CviShape,
    pub fmt: i32,
    pub scale: f32,
    pub zero_point: i32,
    pub ptr: *mut u8,
    pub physical_addr: u64,
}

type FnRegisterModel = unsafe extern "C" fn(*const c_char, *mut CviModelHandle) -> CviRc;
type FnGetInputOutputTensors =
    unsafe extern "C" fn(CviModelHandle, *mut *mut CviTensor, *mut i32, *mut *mut CviTensor, *mut i32) -> CviRc;
type FnForward = unsafe extern "C" fn(CviModelHandle, *mut CviTensor, *mut CviTensor) -> CviRc;
type FnCleanupModel = unsafe extern "C" fn(CviModelHandle) -> CviRc;
type FnTensorPtr = unsafe extern "C" fn(*mut CviTensor) -> *mut u8;

pub struct CviRuntime {
    pub register_model: Symbol<'static, FnRegisterModel>,
    pub get_io_tensors: Symbol<'static, FnGetInputOutputTensors>,
    pub forward: Symbol<'static, FnForward>,
    pub cleanup_model: Symbol<'static, FnCleanupModel>,
    pub tensor_ptr: Symbol<'static, FnTensorPtr>,
}

impl CviRuntime {
    pub fn load() -> Result<Self, String> {
        let candidates = ["libcviruntime.so", "libcviruntime.so.1"];
        let mut errs = vec![];
        for name in &candidates {
            match Self::try_load(name) {
                Ok(rt) => return Ok(rt),
                Err(e) => errs.push(format!("  {name}: {e}")),
            }
        }
        Err(format!(
            "failed to load cviruntime:\n{}\nSet LD_LIBRARY_PATH",
            errs.join("\n")
        ))
    }

    fn try_load(name: &str) -> Result<Self, String> {
        let lib = unsafe { Library::new(name).map_err(|e| e.to_string())? };
        let lib = Box::leak(Box::new(lib));
        unsafe {
            Ok(Self {
                register_model: lib.get(b"CVI_NN_RegisterModel").map_err(|e| e.to_string())?,
                get_io_tensors: lib
                    .get(b"CVI_NN_GetInputOutputTensors")
                    .map_err(|e| e.to_string())?,
                forward: lib.get(b"CVI_NN_Forward").map_err(|e| e.to_string())?,
                cleanup_model: lib.get(b"CVI_NN_CleanupModel").map_err(|e| e.to_string())?,
                tensor_ptr: lib.get(b"CVI_NN_TensorPtr").map_err(|e| e.to_string())?,
            })
        }
    }
}
