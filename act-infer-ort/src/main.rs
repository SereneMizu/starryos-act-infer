use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use ndarray::IxDyn;
use ort::session::Session;
use ort::value::Tensor;

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

    let t = Instant::now();
    let mut session = load_model(&args.model)?;
    let load_ms = t.elapsed().as_millis();
    println!("[ort] model loaded in {load_ms}ms");
    print_mem("after model load");

    let name_a = args.left.file_name().unwrap_or_default().to_string_lossy();
    let t = Instant::now();
    let action_a = run_inference(&mut session, &args.left)?;
    let infer_ms_a = t.elapsed().as_millis();
    println!(
        "[{name_a}] left_vel={:+.6}  right_vel={:+.6}  gripper={:+.6}  infer={infer_ms_a}ms",
        action_a[0], action_a[1], action_a[2]
    );
    let left_ok = action_a[1] > action_a[0];
    println!("  turn: {}", if left_ok { "LEFT (correct)" } else { "WRONG" });
    print_mem("after frame 1");

    let name_b = args.right.file_name().unwrap_or_default().to_string_lossy();
    let t = Instant::now();
    let action_b = run_inference(&mut session, &args.right)?;
    let infer_ms_b = t.elapsed().as_millis();
    println!(
        "[{name_b}] left_vel={:+.6}  right_vel={:+.6}  gripper={:+.6}  infer={infer_ms_b}ms",
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
