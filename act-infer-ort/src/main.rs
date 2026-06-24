use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use ndarray::IxDyn;
use ort::session::Session;
use ort::value::Tensor;
use serde::Deserialize;

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

    fn peak_used_mb(&self, baseline_free_kb: u64) -> u64 {
        let min_free = self.min_free.load(Ordering::Relaxed);
        (baseline_free_kb.saturating_sub(min_free)) / 1024
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
        let raw: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(path).expect("failed to read stats.json"),
        )
        .expect("failed to parse stats.json");

        let parse_arr = |v: &serde_json::Value| -> Vec<f32> {
            v.as_array()
                .expect("expected array")
                .iter()
                .map(|x| x.as_f64().expect("expected number") as f32)
                .collect()
        };

        let state = &raw["observation.state"];
        let action = &raw["action"];

        Self {
            state_q01: parse_arr(&state["q01"]),
            state_q99: parse_arr(&state["q99"]),
            action_q01: parse_arr(&action["q01"]),
            action_q99: parse_arr(&action["q99"]),
        }
    }

    fn normalize_state(&self, state: &[f32]) -> Vec<f32> {
        state
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let d = self.state_q99[i] - self.state_q01[i];
                let d = if d.abs() < 1e-8 { 1e-8 } else { d };
                2.0 * (s - self.state_q01[i]) / d - 1.0
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

#[derive(Parser)]
#[command(name = "act-infer-ort", about = "ACT model ONNX inference via ort")]
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
struct RefEntry {
    frame: String,
    left_vel: f64,
    right_vel: f64,
    #[allow(dead_code)]
    gripper_target: f64,
    turn: String,
}

fn load_reference(path: &Path) -> Vec<RefEntry> {
    let s = std::fs::read_to_string(path).expect("failed to read reference json");
    serde_json::from_str(&s).expect("failed to parse reference json")
}

fn load_model(path: &Path) -> Result<Session, ort::Error> {
    Session::builder()?.commit_from_file(path)
}

fn preprocess_image(path: &Path) -> Result<Tensor<f32>, ort::Error> {
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

fn make_state_tensor(norm: &NormParams) -> Result<Tensor<f32>, ort::Error> {
    let state_dim = norm.state_q01.len();
    let raw_state = vec![0.0f32; state_dim];
    let normalized = norm.normalize_state(&raw_state);
    Tensor::from_array((
        [1usize, state_dim],
        normalized.into_boxed_slice(),
    ))
}

fn collect_frames(dir: &Path) -> Vec<PathBuf> {
    let mut frames: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("failed to read frames directory")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("jpg"))
        })
        .map(|e| e.path())
        .collect();
    frames.sort();
    frames
}

struct FrameResult {
    frame: String,
    img_load_ms: u128,
    infer_ms: u128,
    left_vel: f32,
    right_vel: f32,
    turn: String,
}

fn infer_all(
    session: &mut Session,
    norm: &NormParams,
    frames: &[PathBuf],
    reference: Option<&[RefEntry]>,
    track_mem: bool,
    baseline_free_kb: u64,
) -> ort::Result<(Vec<FrameResult>, Option<u64>)> {
    let tracker = if track_mem {
        Some(MemTracker::new())
    } else {
        None
    };

    let n = frames.len();
    let mut results: Vec<FrameResult> = Vec::with_capacity(n);

    for (i, frame_path) in frames.iter().enumerate() {
        let name = frame_path.file_name().unwrap_or_default().to_string_lossy();

        let t_img = Instant::now();
        let img_tensor = preprocess_image(frame_path)?;
        let img_load_ms = t_img.elapsed().as_millis();

        let state_tensor = make_state_tensor(norm)?;

        let t_infer = Instant::now();
        let outputs = session.run(ort::inputs![img_tensor, state_tensor])?;
        let infer_ms = t_infer.elapsed().as_millis();

        let arr = outputs[0].try_extract_array::<f32>()?;
        let view = arr.view().into_dyn();
        let action_dim = norm.action_q01.len();
        let raw: Vec<f32> = (0..action_dim)
            .map(|d| view[IxDyn(&[0, 0, d])])
            .collect();
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

        let ref_info = reference.as_ref().and_then(|refs| refs.get(i));
        let match_tag = if let Some(ref_e) = ref_info {
            let ref_turn = &ref_e.turn;
            if turn == ref_turn.as_str() {
                "OK"
            } else {
                "DIFF"
            }
        } else {
            ""
        };

        if let Some(ref_e) = ref_info {
            let ref_turn = &ref_e.turn;
            println!(
                "[{name}] img={img_load_ms}ms  infer={infer_ms}ms  left={left_vel:+.6}  right={right_vel:+.6}  turn={turn:<5} ref={ref_turn:<5} [{match_tag}]"
            );
        } else {
            println!(
                "[{name}] img={img_load_ms}ms  infer={infer_ms}ms  left={left_vel:+.6}  right={right_vel:+.6}  turn={turn:<5}"
            );
        }

        if track_mem && (i + 1) % 100 == 0 {
            print_mem(&format!("frame {}/{}", i + 1, n));
        }

        results.push(FrameResult {
            frame: name.into_owned(),
            img_load_ms,
            infer_ms,
            left_vel,
            right_vel,
            turn: turn.to_string(),
        });
    }

    let peak = tracker.as_ref().map(|t| t.peak_used_mb(baseline_free_kb));
    Ok((results, peak))
}

fn print_summary(
    n: usize,
    model_load_ms: u128,
    results: &[FrameResult],
    peak_mb: Option<u64>,
    reference: Option<&[RefEntry]>,
) {
    let infer_times: Vec<u128> = results.iter().map(|r| r.infer_ms).collect();
    let img_times: Vec<u128> = results.iter().map(|r| r.img_load_ms).collect();

    let infer_sum: u128 = infer_times.iter().sum();
    let infer_avg = infer_sum as f64 / n as f64;
    let infer_min = infer_times.iter().min().unwrap();
    let infer_max = infer_times.iter().max().unwrap();

    let img_sum: u128 = img_times.iter().sum();
    let img_avg = img_sum as f64 / n as f64;
    let img_min = img_times.iter().min().unwrap();
    let img_max = img_times.iter().max().unwrap();

    println!("\n===== SUMMARY =====");
    println!("frames:      {n}");
    println!("model load:  {model_load_ms}ms");
    println!(
        "img   total: {}ms  avg: {:.1}ms  min: {img_min}ms  max: {img_max}ms",
        img_sum,
        img_avg,
    );
    println!(
        "infer total: {}ms  avg: {:.1}ms  min: {infer_min}ms  max: {infer_max}ms",
        infer_sum,
        infer_avg,
    );
    if let Some(peak) = peak_mb {
        println!("peak memory: {peak} MB");
    }

    if let Some(ref_entries) = reference {
        let mut turn_match = 0usize;
        let mut turn_differ = 0usize;
        let mut max_left_diff: f32 = 0.0;
        let mut max_right_diff: f32 = 0.0;
        let mut diff_frames: Vec<(&str, &str, f32, f32, &str, f32, f32)> = Vec::new();

        for (r, ref_e) in results.iter().zip(ref_entries.iter()) {
            let l_diff = (r.left_vel - ref_e.left_vel as f32).abs();
            let r_diff = (r.right_vel - ref_e.right_vel as f32).abs();
            if l_diff > max_left_diff {
                max_left_diff = l_diff;
            }
            if r_diff > max_right_diff {
                max_right_diff = r_diff;
            }

            if r.turn == ref_e.turn {
                turn_match += 1;
            } else {
                turn_differ += 1;
                diff_frames.push((
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
        println!("max left_vel diff:  {max_left_diff:.6}");
        println!("max right_vel diff: {max_right_diff:.6}");

        if !diff_frames.is_empty() {
            println!("\n--- Differing frames ({}) ---", diff_frames.len());
            for (name, turn, lv, rv, ref_turn, ref_lv, ref_rv) in &diff_frames {
                println!(
                    "  {name}: rust={turn:<5} (L={lv:+.6} R={rv:+.6})  ref={ref_turn:<5} (L={ref_lv:+.6} R={ref_rv:+.6})"
                );            }
        } else {
            println!("\nAll turn directions match.");
        }
    }

    println!("\nACT_INFER_OK");
}

fn main() -> ort::Result<()> {
    let args = Args::parse();

    if args.track_mem {
        print_mem("startup");
    }
    let baseline_free_kb = read_mem_free_kb();

    let stats_path = args.stats;
    let norm = NormParams::load(&stats_path);
    println!(
        "[stats] state_dim={}  action_dim={}",
        norm.state_q01.len(),
        norm.action_q01.len()
    );

    let t = Instant::now();
    let mut session = load_model(&args.model)?;
    let load_ms = t.elapsed().as_millis();
    println!("[ort] model loaded in {load_ms}ms");
    if args.track_mem {
        print_mem("after model load");
    }

    let frames = collect_frames(&args.dir);
    let n = frames.len();
    println!("[infer] {n} frames in {}", args.dir.display());

    let reference = args.reference.as_ref().map(|p| load_reference(p));
    if let Some(ref_entries) = &reference {
        println!("[verify] reference: {} frames", ref_entries.len());
    }

    let (results, peak_mb) = infer_all(&mut session, &norm, &frames, reference.as_deref(), args.track_mem, baseline_free_kb)?;

    if args.track_mem {
        print_mem("after all frames");
    }
    print_summary(n, load_ms, &results, peak_mb, reference.as_deref());

    Ok(())
}
