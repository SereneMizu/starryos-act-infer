mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use serde::Deserialize;

use bindings::{CVI_FMT, CVI_MODEL_HANDLE, CVI_RC, CVI_TENSOR};

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

struct NormParams {
    state_q01: Vec<f32>,
    state_q99: Vec<f32>,
    action_q01: Vec<f32>,
    action_q99: Vec<f32>,
}

impl NormParams {
    fn load(path: &Path) -> Self {
        let raw: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).expect("stats.json not found"))
                .expect("invalid stats.json");

        let parse = |v: &serde_json::Value| -> Vec<f32> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        };

        let state = &raw["observation.state"];
        let action = &raw["action"];
        Self {
            state_q01: parse(&state["q01"]),
            state_q99: parse(&state["q99"]),
            action_q01: parse(&action["q01"]),
            action_q99: parse(&action["q99"]),
        }
    }

    fn normalize_state(&self, raw: &[f32]) -> Vec<f32> {
        raw.iter()
            .enumerate()
            .map(|(i, &v)| {
                let d = self.state_q99[i] - self.state_q01[i];
                let d = if d.abs() < 1e-8 { 1e-8 } else { d };
                2.0 * (v - self.state_q01[i]) / d - 1.0
            })
            .collect()
    }

    fn denorm_action(&self, raw: &[f32]) -> Vec<f32> {
        raw.iter()
            .enumerate()
            .map(|(i, &v)| {
                let d = self.action_q99[i] - self.action_q01[i];
                let d = if d.abs() < 1e-8 { 1e-8 } else { d };
                (v + 1.0) / 2.0 * d + self.action_q01[i]
            })
            .collect()
    }
}

fn preprocess_image(path: &Path) -> Vec<f32> {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("failed to open {}: {e}", path.display()))
        .resize_exact(224, 224, image::imageops::FilterType::Triangle)
        .to_rgb8();

    let mut data = vec![0.0f32; 3 * 224 * 224];
    for y in 0..224 {
        for x in 0..224 {
            let px = img.get_pixel(x as u32, y as u32);
            for c in 0..3usize {
                let val = px[c] as f32 / 255.0;
                data[c * 224 * 224 + y * 224 + x] = (val - MEAN[c]) / STD[c];
            }
        }
    }
    data
}

fn build_images_tensor(image_data: &[f32]) -> Vec<f32> {
    let mut t = vec![0.0f32; 1 * 1 * 3 * 224 * 224];
    for c in 0..3 {
        for y in 0..224 {
            for x in 0..224 {
                let idx = c * 224 * 224 + y * 224 + x;
                t[idx] = image_data[idx];
            }
        }
    }
    t
}

struct TpuModel {
    handle: CVI_MODEL_HANDLE,
    inputs: *mut CVI_TENSOR,
    input_num: i32,
    outputs: *mut CVI_TENSOR,
    output_num: i32,
}

impl TpuModel {
    fn load(path: &Path) -> Result<Self, String> {
        let c_path = std::ffi::CString::new(path.to_string_lossy().as_bytes())
            .map_err(|e| format!("invalid path: {e}"))?;

        let mut handle: CVI_MODEL_HANDLE = ptr::null_mut();
        let rc: CVI_RC = unsafe { bindings::CVI_NN_RegisterModel(c_path.as_ptr(), &mut handle) };
        if rc != 0 {
            return Err(format!("CVI_NN_RegisterModel failed: rc={rc}"));
        }
        let mut inputs: *mut CVI_TENSOR = ptr::null_mut();
        let mut input_num: i32 = 0;
        let mut outputs: *mut CVI_TENSOR = ptr::null_mut();
        let mut output_num: i32 = 0;

        let rc: CVI_RC = unsafe {
            bindings::CVI_NN_GetInputOutputTensors(
                handle, &mut inputs, &mut input_num, &mut outputs, &mut output_num,
            )
        };
        if rc != 0 {
            unsafe { bindings::CVI_NN_CleanupModel(handle) };
            return Err(format!("CVI_NN_GetInputOutputTensors failed: rc={rc}"));
        }

        Ok(Self { handle, inputs, input_num, outputs, output_num })
    }

    fn input_shape(&self, idx: isize) -> &[i32] {
        let t = unsafe { &*self.inputs.offset(idx) };
        &t.shape.dim[..t.shape.dim_size as usize]
    }

    fn output_shape(&self, idx: isize) -> &[i32] {
        let t = unsafe { &*self.outputs.offset(idx) };
        &t.shape.dim[..t.shape.dim_size as usize]
    }

    fn output_fmt(&self, idx: isize) -> CVI_FMT {
        unsafe { (*self.outputs.offset(idx)).fmt }
    }

    fn input_ptr(&self, idx: isize) -> *mut std::ffi::c_void {
        unsafe { bindings::CVI_NN_TensorPtr(self.inputs.offset(idx)) }
    }

    fn output_ptr(&self, idx: isize) -> *mut std::ffi::c_void {
        unsafe { bindings::CVI_NN_TensorPtr(self.outputs.offset(idx)) }
    }

    fn run(&self) -> Result<(), String> {
        let rc: CVI_RC = unsafe {
            bindings::CVI_NN_Forward(
                self.handle, self.inputs, self.input_num, self.outputs, self.output_num,
            )
        };
        if rc != 0 {
            return Err(format!("CVI_NN_Forward failed: rc={rc}"));
        }
        Ok(())
    }
}

impl Drop for TpuModel {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { bindings::CVI_NN_CleanupModel(self.handle) };
        }
    }
}

#[derive(Parser)]
#[command(name = "act-infer-tpu", about = "ACT model TPU inference on SG2002")]
struct Args {
    #[arg(short, long)]
    model: PathBuf,

    #[arg(short, long)]
    dir: PathBuf,

    #[arg(short, long)]
    stats: PathBuf,

    #[arg(short, long)]
    reference: Option<PathBuf>,

    #[arg(long)]
    track_mem: bool,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct RefEntry {
    frame: String,
    left_vel: f64,
    right_vel: f64,
    turn: String,
}

fn load_reference(path: &Path) -> Vec<RefEntry> {
    let s = std::fs::read_to_string(path).expect("reference file not found");
    serde_json::from_str(&s).expect("invalid reference json")
}

struct FrameResult {
    frame: String,
    infer_ms: u128,
    left_vel: f32,
    right_vel: f32,
    turn: String,
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn read_fp32_from_tensor(ptr: *const u8, len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; len];
    unsafe { ptr::copy_nonoverlapping(ptr as *const f32, out.as_mut_ptr(), len) }
    out
}

fn read_bf16_from_tensor(ptr: *const u8, len: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(len);
    let bf16_ptr = ptr as *const u16;
    for i in 0..len {
        let bits = unsafe { *bf16_ptr.add(i) };
        out.push(bf16_to_f32(bits));
    }
    out
}

fn read_mem_free_kb() -> u64 {
    let s = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    for line in s.lines() {
        if line.starts_with("MemFree:") {
            return line.split_whitespace().nth(1).unwrap_or("0").parse().unwrap_or(0);
        }
    }
    0
}

fn print_mem(tag: &str) {
    let s = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let mut mem_total = "";
    let mut mem_free = "";
    let mut mem_avail = "";
    for line in s.lines() {
        if line.starts_with("MemTotal:") {
            mem_total = line;
        } else if line.starts_with("MemFree:") {
            mem_free = line;
        } else if line.starts_with("MemAvailable:") {
            mem_avail = line;
        }
    }
    println!("[mem] {tag}: {mem_total}  {mem_free}  {mem_avail}");
}

struct MemTracker {
    min_free: Arc<AtomicU64>,
}

impl MemTracker {
    fn new() -> Self {
        let min_free = Arc::new(AtomicU64::new(u64::MAX));
        let min_free_clone = min_free.clone();
        std::thread::spawn(move || loop {
            let free = read_mem_free_kb();
            min_free_clone.fetch_min(free, Ordering::Relaxed);
            std::thread::sleep(std::time::Duration::from_millis(10));
        });
        Self { min_free }
    }

    fn peak_used_mb(&self, baseline_free_kb: u64) -> u64 {
        let min_free = self.min_free.load(Ordering::Relaxed);
        (baseline_free_kb.saturating_sub(min_free)) / 1024
    }
}

fn main() {
    let args = Args::parse();

    if args.track_mem {
        print_mem("startup");
    }
    let baseline_free_kb = read_mem_free_kb();
    let tracker = if args.track_mem {
        Some(MemTracker::new())
    } else {
        None
    };

    let norm = NormParams::load(&args.stats);
    println!(
        "[stats] state_dim={}  action_dim={}",
        norm.state_q01.len(),
        norm.action_q01.len()
    );

    let t = Instant::now();
    let model = TpuModel::load(&args.model).expect("failed to load cvimodel");
    let load_ms = t.elapsed().as_millis();
    println!("[tpu] model loaded in {load_ms}ms");

    let img_shape = model.input_shape(0);
    let state_shape = model.input_shape(1);
    let action_shape = model.output_shape(0);
    let output_fmt = model.output_fmt(0);
    println!(
        "[tpu] input:{} images={:?} state={:?}  output:{} action={:?} fmt={:?}",
        model.input_num, img_shape, state_shape,
        model.output_num, action_shape, output_fmt,
    );

    let img_tensor_ptr = model.input_ptr(0);
    let state_tensor_ptr = model.input_ptr(1);
    let out_tensor_ptr = model.output_ptr(0);
    println!(
        "[tpu] tensor ptrs  images={img_tensor_ptr:p}  state={state_tensor_ptr:p}  output={out_tensor_ptr:p}"
    );
    if img_tensor_ptr.is_null() || state_tensor_ptr.is_null() || out_tensor_ptr.is_null() {
        panic!("NULL tensor pointer");
    }

    if args.track_mem {
        print_mem("after model load");
    }

    let state_dim = state_shape[1] as usize;
    let raw_state = vec![0.0f32; state_dim];
    let normalized_state = norm.normalize_state(&raw_state);

    let frames = collect_frames(&args.dir);
    let n = frames.len();
    println!("[infer] {n} frames in {}", args.dir.display());

    let reference = args.reference.as_ref().map(|p| load_reference(p));
    if let Some(ref_entries) = &reference {
        println!("[verify] reference: {} frames", ref_entries.len());
    }

    let action_dim = norm.action_q01.len();
    let mut results: Vec<FrameResult> = Vec::with_capacity(n);

    for (i, frame_path) in frames.iter().enumerate() {
        let name = frame_path.file_name().unwrap_or_default().to_string_lossy().to_string();

        let t_infer = Instant::now();
        let img_data = preprocess_image(frame_path);
        let img_tensor = build_images_tensor(&img_data);

        let inp0 = model.input_ptr(0) as *mut f32;
        let inp1 = model.input_ptr(1) as *mut f32;

        unsafe {
            ptr::copy_nonoverlapping(img_tensor.as_ptr(), inp0, img_tensor.len());
            ptr::copy_nonoverlapping(normalized_state.as_ptr(), inp1, normalized_state.len());
        }

        model.run().expect("CVI_NN_Forward failed");
        let infer_ms = t_infer.elapsed().as_millis();

        let out0 = model.output_ptr(0) as *const u8;
        let raw = match output_fmt {
            CVI_FMT::CVI_FMT_BF16 => read_bf16_from_tensor(out0, action_dim),
            _ => read_fp32_from_tensor(out0, action_dim),
        };
        let action = norm.denorm_action(&raw);

        let left_vel = action[0];
        let right_vel = action[1];
        let turn = if left_vel < right_vel {
            "LEFT"
        } else if left_vel > right_vel {
            "RIGHT"
        } else {
            "STRAIGHT"
        };

        let ref_info = reference.as_ref().and_then(|r| r.get(i));
        let match_tag = match ref_info {
            Some(r) if turn == r.turn.as_str() => "OK",
            Some(_) => "DIFF",
            None => "",
        };

        if let Some(r) = ref_info {
            println!(
                "[{name}] infer={infer_ms}ms  left={left_vel:+.6}  right={right_vel:+.6}  turn={turn:<5} ref={ref_turn:<5} [{match_tag}]",
                ref_turn = r.turn,
            );
        } else {
            println!(
                "[{name}] infer={infer_ms}ms  left={left_vel:+.6}  right={right_vel:+.6}  turn={turn:<5}"
            );
        }

        if args.track_mem && (i + 1) % 100 == 0 {
            print_mem(&format!("frame {}/{}", i + 1, n));
        }

        results.push(FrameResult {
            frame: name,
            infer_ms,
            left_vel,
            right_vel,
            turn: turn.to_string(),
        });
    }

    if args.track_mem {
        print_mem("after all frames");
    }

    print_summary(n, &results, tracker.as_ref().map(|t| t.peak_used_mb(baseline_free_kb)), reference.as_deref());
}

fn print_summary(
    n: usize,
    results: &[FrameResult],
    peak_mb: Option<u64>,
    reference: Option<&[RefEntry]>,
) {
    let times: Vec<u128> = results.iter().map(|r| r.infer_ms).collect();
    let sum: u128 = times.iter().sum();
    let avg = sum as f64 / n as f64;
    let min = times.iter().min().unwrap();
    let max = times.iter().max().unwrap();

    println!("\n===== SUMMARY =====");
    println!("frames:      {n}");
    println!(
        "infer total: {:.2}s  avg: {:.1}ms  min: {min}ms  max: {max}ms",
        sum as f64 / 1000.0,
        avg,
    );
    if let Some(peak) = peak_mb {
        println!("peak memory: {peak} MB");
    }

    if let Some(refs) = reference {
        let mut turn_match = 0;
        let mut turn_differ = 0;
        let mut max_l_diff: f32 = 0.0;
        let mut max_r_diff: f32 = 0.0;
        let mut diffs: Vec<(&str, &str, f32, f32, &str, f32, f32)> = Vec::new();

        for (r, ref_e) in results.iter().zip(refs.iter()) {
            let ld = (r.left_vel - ref_e.left_vel as f32).abs();
            let rd = (r.right_vel - ref_e.right_vel as f32).abs();
            if ld > max_l_diff { max_l_diff = ld; }
            if rd > max_r_diff { max_r_diff = rd; }

            if r.turn == ref_e.turn {
                turn_match += 1;
            } else {
                turn_differ += 1;
                diffs.push((&r.frame, &r.turn, r.left_vel, r.right_vel, &ref_e.turn, ref_e.left_vel as f32, ref_e.right_vel as f32));
            }
        }

        println!("\n===== VERIFY =====");
        println!(
            "turn match:  {turn_match}/{n} ({:.1}%)",
            turn_match as f64 / n as f64 * 100.0
        );
        println!("turn differ: {turn_differ}/{n}");
        println!("max left_vel diff:  {max_l_diff:.6}");
        println!("max right_vel diff: {max_r_diff:.6}");

        if !diffs.is_empty() {
            println!("\n--- Differing frames ({}) ---", diffs.len());
            for (name, turn, lv, rv, ref_turn, ref_lv, ref_rv) in &diffs {
                println!(
                    "  {name}: tpu={turn:<5} (L={lv:+.6} R={rv:+.6})  ref={ref_turn:<5} (L={ref_lv:+.6} R={ref_rv:+.6})"
                );
            }
        } else {
            println!("\nAll turn directions match.");
        }
    }

    println!("\nACT_INFER_OK");
}

fn collect_frames(dir: &Path) -> Vec<PathBuf> {
    let mut frames: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("frames dir not found")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x.eq_ignore_ascii_case("jpg")))
        .map(|e| e.path())
        .collect();
    frames.sort();
    frames
}
