mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use serde::Deserialize;

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

unsafe fn zeroed<T>() -> T {
    unsafe { MaybeUninit::<T>::zeroed().assume_init() }
}

struct RknnModel {
    ctx: bindings::rknn_context,
    n_input: u32,
    n_output: u32,
    input_attrs: Vec<bindings::rknn_tensor_attr>,
    output_attrs: Vec<bindings::rknn_tensor_attr>,
}

impl RknnModel {
    fn load(path: &Path) -> Result<Self, String> {
        let c_path = std::ffi::CString::new(path.to_string_lossy().as_bytes())
            .map_err(|e| format!("invalid path: {e}"))?;

        let mut ctx: bindings::rknn_context = 0;
        // size=0 表示 model 为文件路径
        let rc = unsafe {
            bindings::rknn_init(
                &mut ctx,
                c_path.as_ptr() as *mut c_void,
                0,
                0,
                ptr::null_mut(),
            )
        };
        if rc != bindings::RKNN_SUCC as i32 {
            return Err(format!("rknn_init failed: rc={rc}"));
        }

        let mut num: bindings::rknn_input_output_num = unsafe { zeroed() };
        let rc = unsafe {
            bindings::rknn_query(
                ctx,
                bindings::_rknn_query_cmd::RKNN_QUERY_IN_OUT_NUM,
                &mut num as *mut _ as *mut c_void,
                std::mem::size_of::<bindings::rknn_input_output_num>() as u32,
            )
        };
        if rc != 0 {
            unsafe { bindings::rknn_destroy(ctx) };
            return Err(format!("rknn_query IN_OUT_NUM failed: rc={rc}"));
        }

        let mut input_attrs = Vec::with_capacity(num.n_input as usize);
        for i in 0..num.n_input {
            let mut attr: bindings::rknn_tensor_attr = unsafe { zeroed() };
            attr.index = i;
            let rc = unsafe {
                bindings::rknn_query(
                    ctx,
                    bindings::_rknn_query_cmd::RKNN_QUERY_INPUT_ATTR,
                    &mut attr as *mut _ as *mut c_void,
                    std::mem::size_of::<bindings::rknn_tensor_attr>() as u32,
                )
            };
            if rc != 0 {
                unsafe { bindings::rknn_destroy(ctx) };
                return Err(format!("rknn_query INPUT_ATTR[{i}] failed: rc={rc}"));
            }
            input_attrs.push(attr);
        }

        let mut output_attrs = Vec::with_capacity(num.n_output as usize);
        for i in 0..num.n_output {
            let mut attr: bindings::rknn_tensor_attr = unsafe { zeroed() };
            attr.index = i;
            let rc = unsafe {
                bindings::rknn_query(
                    ctx,
                    bindings::_rknn_query_cmd::RKNN_QUERY_OUTPUT_ATTR,
                    &mut attr as *mut _ as *mut c_void,
                    std::mem::size_of::<bindings::rknn_tensor_attr>() as u32,
                )
            };
            if rc != 0 {
                unsafe { bindings::rknn_destroy(ctx) };
                return Err(format!("rknn_query OUTPUT_ATTR[{i}] failed: rc={rc}"));
            }
            output_attrs.push(attr);
        }

        Ok(Self {
            ctx,
            n_input: num.n_input,
            n_output: num.n_output,
            input_attrs,
            output_attrs,
        })
    }

    fn input_shape(&self, idx: usize) -> &[u32] {
        let a = &self.input_attrs[idx];
        &a.dims[..a.n_dims as usize]
    }

    fn output_shape(&self, idx: usize) -> &[u32] {
        let a = &self.output_attrs[idx];
        &a.dims[..a.n_dims as usize]
    }
}

impl Drop for RknnModel {
    fn drop(&mut self) {
        unsafe { bindings::rknn_destroy(self.ctx) };
    }
}

#[derive(Parser)]
#[command(name = "act-infer-rknn", about = "ACT model RKNN inference on RK3588")]
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

fn read_self_vm_kb() -> u64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in s.lines() {
        if line.starts_with("VmSize:") {
            return line
                .split_whitespace()
                .nth(1)
                .and_then(|x| x.parse().ok())
                .unwrap_or(0);
        }
    }
    0
}

fn print_mem(tag: &str) {
    let vm = read_self_vm_kb();
    let s = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let free: u64 = s
        .lines()
        .find(|l| l.starts_with("MemFree:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|x| x.parse().ok())
        .unwrap_or(0);
    println!("[mem] {tag}: VmSize={vm}kB  MemFree={free}kB");
}

struct MemTracker {
    peak_vm_kb: Arc<AtomicU64>,
}

impl MemTracker {
    fn new() -> Self {
        let peak_vm_kb = Arc::new(AtomicU64::new(0));
        let c = peak_vm_kb.clone();
        std::thread::spawn(move || loop {
            let vm = read_self_vm_kb();
            let prev = c.load(Ordering::Relaxed);
            if vm > prev {
                c.store(vm, Ordering::Relaxed);
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        });
        Self { peak_vm_kb }
    }

    fn peak_mb(&self) -> u64 {
        self.peak_vm_kb.load(Ordering::Relaxed) / 1024
    }
}

fn main() {
    let args = Args::parse();

    if args.track_mem {
        print_mem("startup");
    }
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
    let model = RknnModel::load(&args.model).expect("failed to load rknn model");
    let load_ms = t.elapsed().as_millis();
    println!("[rknn] model loaded in {load_ms}ms");

    let img_shape = model.input_shape(0);
    let state_shape = model.input_shape(1);
    let action_shape = model.output_shape(0);
    println!(
        "[rknn] input:{} images={:?} state={:?}  output:{} action={:?}",
        model.n_input, img_shape, state_shape,
        model.n_output, action_shape,
    );

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
        let name = frame_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        let t_infer = Instant::now();
        let img_data = preprocess_image(frame_path);
        let img_tensor = build_images_tensor(&img_data);

        let mut inputs: [bindings::rknn_input; 2] = unsafe { zeroed() };
        for inp in inputs.iter_mut() {
            inp.type_ = bindings::_rknn_tensor_type::RKNN_TENSOR_FLOAT32;
            inp.fmt = bindings::_rknn_tensor_format::RKNN_TENSOR_NCHW;
        }
        inputs[0].index = 0;
        inputs[0].buf = img_tensor.as_ptr() as *mut c_void;
        inputs[0].size = (img_tensor.len() * 4) as u32;
        inputs[1].index = 1;
        inputs[1].buf = normalized_state.as_ptr() as *mut c_void;
        inputs[1].size = (normalized_state.len() * 4) as u32;

        let rc = unsafe { bindings::rknn_inputs_set(model.ctx, 2, inputs.as_mut_ptr()) };
        if rc != 0 {
            panic!("rknn_inputs_set failed: rc={rc}");
        }

        let rc = unsafe { bindings::rknn_run(model.ctx, ptr::null_mut()) };
        if rc != 0 {
            panic!("rknn_run failed: rc={rc}");
        }

        let mut outputs: [bindings::rknn_output; 1] = unsafe { zeroed() };
        outputs[0].want_float = 1;
        outputs[0].is_prealloc = 0;
        let rc = unsafe {
            bindings::rknn_outputs_get(model.ctx, 1, outputs.as_mut_ptr(), ptr::null_mut())
        };
        if rc != 0 {
            panic!("rknn_outputs_get failed: rc={rc}");
        }

        // want_float=1: 输出已转为 fp32；取首个时间步 [left_vel, right_vel, gripper]
        let out_ptr = outputs[0].buf as *const f32;
        let raw: Vec<f32> = (0..action_dim)
            .map(|k| unsafe { *out_ptr.add(k) })
            .collect();
        let infer_ms = t_infer.elapsed().as_millis();

        unsafe { bindings::rknn_outputs_release(model.ctx, 1, outputs.as_mut_ptr()) };

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

    print_summary(
        n,
        &results,
        tracker.as_ref().map(|t| t.peak_mb()),
        reference.as_deref(),
    );
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
            if ld > max_l_diff {
                max_l_diff = ld;
            }
            if rd > max_r_diff {
                max_r_diff = rd;
            }

            if r.turn == ref_e.turn {
                turn_match += 1;
            } else {
                turn_differ += 1;
                diffs.push((
                    &r.frame,
                    &r.turn,
                    r.left_vel,
                    r.right_vel,
                    &ref_e.turn,
                    ref_e.left_vel as f32,
                    ref_e.right_vel as f32,
                ));
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
                    "  {name}: rknn={turn:<5} (L={lv:+.6} R={rv:+.6})  ref={ref_turn:<5} (L={ref_lv:+.6} R={ref_rv:+.6})"
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
