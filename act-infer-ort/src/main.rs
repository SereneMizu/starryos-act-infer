use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use ndarray::IxDyn;
use ort::session::Session;
use ort::value::Tensor;

fn read_mem_free_kb() -> u64 {
    let s = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    for line in s.lines() {
        if line.starts_with("MemFree:") {
            return line.split_whitespace().nth(1).unwrap_or("0").parse().unwrap_or(0);
        }
    }
    0
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

    fn peak_used_mb(&self, total_kb: u64) -> u64 {
        let min_free = self.min_free.load(Ordering::Relaxed);
        (total_kb.saturating_sub(min_free)) / 1024
    }

    fn reset(&self) {
        self.min_free.store(u64::MAX, Ordering::Relaxed);
    }
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

const STATE_NORM: [f32; 2] = [-0.433693, -1.0];
const ACTION_Q01: [f32; 3] = [-0.1, 0.0, 0.0];
const ACTION_D: [f32; 3] = [0.3, 0.2, 0.0];
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Parser)]
#[command(name = "act-infer-ort", about = "ACT model ONNX inference via ort")]
struct Args {
    #[arg(short, long)]
    model: PathBuf,

    #[arg(long)]
    left: PathBuf,

    #[arg(long)]
    right: PathBuf,
}

fn load_model(path: &PathBuf) -> Result<Session, ort::Error> {
    Session::builder()?.commit_from_file(path)
}

fn preprocess_image(path: &PathBuf) -> Result<Tensor<f32>, ort::Error> {
    let img = image::open(path)
        .map_err(|e| ort::Error::new(format!("image open failed: {e}")))?
        .resize_exact(224, 224, image::imageops::FilterType::Triangle);
    let rgb = img.to_rgb8();

    let shape: Vec<usize> = vec![1, 1, 3, 224, 224];
    let mut data = vec![0.0f32; 1 * 1 * 3 * 224 * 224];

    for y in 0..224usize {
        for x in 0..224usize {
            let px = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3usize {
                let val = px[c] as f32 / 255.0;
                data[c * 224 * 224 + y * 224 + x] = (val - MEAN[c]) / STD[c];
            }
        }
    }

    Tensor::from_array((shape, data.into_boxed_slice()))
}

fn make_state_tensor() -> Result<Tensor<f32>, ort::Error> {
    let state: Vec<f32> = STATE_NORM.to_vec();
    Tensor::from_array(([1usize, 2], state.into_boxed_slice()))
}

fn denorm(val: f32, dim: usize) -> f32 {
    if ACTION_D[dim].abs() < 1e-8 {
        return ACTION_Q01[dim];
    }
    (val + 1.0) / 2.0 * ACTION_D[dim] + ACTION_Q01[dim]
}

fn run_inference(session: &mut Session, image_path: &PathBuf) -> Result<[f32; 3], ort::Error> {
    let img_tensor = preprocess_image(image_path)?;
    let state_tensor = make_state_tensor()?;

    let outputs = session.run(ort::inputs![img_tensor, state_tensor])?;

    let arr = outputs[0].try_extract_array::<f32>()?;
    let view = arr.view().into_dyn();

    Ok([
        denorm(view[IxDyn(&[0, 0, 0])], 0),
        denorm(view[IxDyn(&[0, 0, 1])], 1),
        denorm(view[IxDyn(&[0, 0, 2])], 2),
    ])
}

fn main() -> ort::Result<()> {
    let args = Args::parse();

    print_mem("startup");
    let total_kb = read_mem_free_kb();
    let tracker = MemTracker::new();
    let startup_free = read_mem_free_kb();

    let t = Instant::now();
    let mut session = load_model(&args.model)?;
    let load_ms = t.elapsed().as_millis();
    let peak_load = tracker.peak_used_mb(total_kb);
    println!("[ort] model loaded in {load_ms}ms  peak memory during load: {peak_load} MB");
    print_mem("after model load");
    tracker.reset();

    let name_a = args.left.file_name().unwrap_or_default().to_string_lossy();
    let t = Instant::now();
    let action_a = run_inference(&mut session, &args.left)?;
    let infer_ms_a = t.elapsed().as_millis();
    let peak_infer_a = tracker.peak_used_mb(total_kb);
    println!(
        "[{name_a}] left_vel={:+.6}  right_vel={:+.6}  gripper={:+.6}  infer={infer_ms_a}ms  peak={peak_infer_a} MB",
        action_a[0], action_a[1], action_a[2]
    );
    let left_ok = action_a[1] > action_a[0];
    println!("  turn: {}", if left_ok { "LEFT (correct)" } else { "WRONG" });
    print_mem("after frame 1");
    tracker.reset();

    let name_b = args.right.file_name().unwrap_or_default().to_string_lossy();
    let t = Instant::now();
    let action_b = run_inference(&mut session, &args.right)?;
    let infer_ms_b = t.elapsed().as_millis();
    let peak_infer_b = tracker.peak_used_mb(total_kb);
    println!(
        "[{name_b}] left_vel={:+.6}  right_vel={:+.6}  gripper={:+.6}  infer={infer_ms_b}ms  peak={peak_infer_b} MB",
        action_b[0], action_b[1], action_b[2]
    );
    let right_ok = action_b[0] > action_b[1];
    println!("  turn: {}", if right_ok { "RIGHT (correct)" } else { "WRONG" });
    print_mem("after frame 2");

    if left_ok && right_ok {
        println!("\nACT_INFER_OK");
    } else {
        eprintln!("\nACT_INFER_FAIL");
        std::process::exit(1);
    }

    Ok(())
}
