use chrono::{NaiveDateTime, Utc};
use clap::Parser;
use eframe::egui;
use egui::{epaint, Stroke};
use egui_plot::{Axis, Legend, PlotPoint, PlotPoints, Points};
use egui_plot::{Line, Plot};
use itertools::Itertools;
use ndarray::s;
use sqlx::postgres::PgPoolOptions;
use sqlx::Row;
use std::borrow::Cow;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
// use sqlx::types::JsonValue
use std::collections::HashMap;
use std::collections::HashSet;
use std::env;
use std::ops::RangeInclusive;
use std::sync::mpsc::{self, Receiver, Sender};

pub mod np;
use colorous::CIVIDIS;
use ndarray_stats::QuantileExt;
use np::load_npy_bytes;

#[derive(Parser, Debug)]
struct Args {
    #[arg(default_value = "../../")]
    artifacts: String,
}

#[derive(Default, Debug, Clone)]
struct Metric {
    xaxis: String,
    orig_values: Vec<[f64; 3]>,
    values: Vec<[f64; 2]>,
    resampled: Vec<[f64; 2]>,
    last_value: chrono::NaiveDateTime,
}

#[derive(Default, Debug, Clone)]
struct Run {
    params: HashMap<String, String>,
    artifacts: HashMap<String, ArtifactId>,
    metrics: HashMap<String, Metric>,
    created_at: chrono::NaiveDateTime,
    // last_update:
}

#[derive(Default, Debug, Clone)]
struct Runs {
    runs: HashMap<String, Run>,
    active_runs: Vec<String>, // filtered_runs: HashMap<String, Run>,
    active_runs_time_ordered: Vec<(String, chrono::NaiveDateTime)>,
    time_filtered_runs: Vec<String>,
}

#[derive(Default, Debug, Clone)]
enum XAxis {
    #[default]
    Batch,
    Time,
}

#[derive(Default, Debug, Clone)]
struct GuiParams {
    n_average: usize,
    max_n: usize,
    param_filters: HashMap<String, HashSet<String>>,
    param_name_filter: String,
    metric_filters: HashSet<String>,
    artifact_filters: HashSet<String>,
    inspect_params: HashSet<String>,
    time_filter_idx: usize,
    time_filter: Option<chrono::NaiveDateTime>,
    x_axis: XAxis,
}

#[derive(PartialEq, Eq)]
enum DataStatus {
    Waiting,
    FirstDataArrived,
    FirstDataProcessed,
    FirstDataPlotted,
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Clone)]
struct ArtifactId {
    artifact_id: i32,
    train_id: String,
    name: String,
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Clone)]
struct NPYArtifactView {
    artifact_id: ArtifactId,
    index: Vec<usize>,
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Clone)]
struct NPYTabularArtifactView {
    artifact_id: ArtifactId,
    x: usize,
    y: usize,
    c: usize,
    log_x: bool,
    log_y: bool, // index: Vec<usize>,
}

enum NPYArray {
    Loading(BinaryArtifact),
    Loaded(ndarray::ArrayD<f32>),
    Error(String),
}

#[derive(Clone)]
struct DownloadProgressStatus {
    downloaded: usize,
    size: usize,
}

struct DownloadProgress {
    rx_update: Receiver<DownloadProgressStatus>,
    status: DownloadProgressStatus,
}

enum BinaryArtifact {
    Loading((JoinHandle<Result<Vec<u8>, String>>, DownloadProgress)),
    Loaded(Vec<u8>),
    Error(String),
}

fn download_artifact(
    artifact_id: ArtifactId,
    tx_path_mutex: Arc<Mutex<Sender<i32>>>,
    rx_artifact_mutex: Arc<Mutex<Receiver<ArtifactTransfer>>>,
) -> BinaryArtifact {
    let (tx_update, rx_update) = mpsc::channel::<DownloadProgressStatus>();
    let join_handle = tokio::spawn(async move {
        let tx_db_artifact_path = tx_path_mutex.lock_owned().await;
        println!(
            "[db] Requesting download {:?}",
            artifact_id // train_id.clone(),
                        // name.clone()
        );
        tx_db_artifact_path.send(artifact_id.artifact_id);
        println!("[db] Waiting for download {:?}", artifact_id);
        let rx_db_artifact = rx_artifact_mutex.lock_owned().await;
        loop {
            let rx_res = rx_db_artifact.recv();
            // println!("[db] recv {:?}", rx_res);
            match rx_res {
                Ok(artifact_binary_res) => match artifact_binary_res {
                    ArtifactTransfer::Done(artifact_binary) => {
                        return Ok(artifact_binary);
                    }
                    ArtifactTransfer::Err(artifact_binary_err) => {
                        return Err(artifact_binary_err.to_string());
                    }
                    ArtifactTransfer::Loading(downloaded, size) => {
                        tx_update.send(DownloadProgressStatus { downloaded, size });
                    }
                },
                Err(err) => {
                    return Err(err.to_string());
                }
            }
        }
    });
    BinaryArtifact::Loading((
        join_handle,
        DownloadProgress {
            rx_update,
            status: DownloadProgressStatus {
                downloaded: 0,
                size: 0,
            },
        },
    ))
}

fn poll_artifact_download(binary_artifact: &mut BinaryArtifact) {
    let mut new_binary_artifact: Option<BinaryArtifact> = None;
    match binary_artifact {
        BinaryArtifact::Loading((join_handle, download_progress)) => {
            // println!("[ARTIFACTS] Loading ....");
            if join_handle.is_finished() {
                let data_res = tokio::runtime::Handle::current()
                    .block_on(join_handle)
                    .unwrap();
                match data_res {
                    Ok(data) => {
                        new_binary_artifact = Some(BinaryArtifact::Loaded(data));
                    }
                    Err(err) => new_binary_artifact = Some(BinaryArtifact::Error(err.to_string())),
                }
            } else if let Ok(download_status) = download_progress.rx_update.try_recv() {
                download_progress.status = download_status;
            }
        }
        BinaryArtifact::Loaded(_) => {}
        BinaryArtifact::Error(_) => {}
    }
    if let Some(new_binary_artifact) = new_binary_artifact {
        *binary_artifact = new_binary_artifact;
    }
}

enum ArtifactHandler {
    NPYHealPixArtifact {
        textures: HashMap<NPYArtifactView, egui::TextureHandle>,
        arrays: HashMap<ArtifactId, NPYArray>,
        views: HashMap<ArtifactId, NPYArtifactView>,
    },
    NPYTabularArtifact {
        arrays: HashMap<ArtifactId, NPYArray>,
        views: HashMap<ArtifactId, NPYTabularArtifactView>,
    },
    ImageArtifact {
        // images: HashMap<ArtifactId, String>,
        binaries: HashMap<ArtifactId, BinaryArtifact>,
    },
}

fn add_artifact(
    handler: &mut ArtifactHandler,
    artifact_id: &ArtifactId,
    args: &Args,
    tx_path_mutex: &mut Arc<Mutex<Sender<i32>>>,
    rx_artifact_mutex: &Arc<Mutex<Receiver<ArtifactTransfer>>>,
) {
    match handler {
        ArtifactHandler::NPYHealPixArtifact {
            arrays,
            textures: _,
            views: _,
        } => handle_add_npy(arrays, &artifact_id, tx_path_mutex, rx_artifact_mutex),
        ArtifactHandler::ImageArtifact { binaries } => match binaries.get_mut(&artifact_id) {
            Some(binary_artifact) => {
                poll_artifact_download(binary_artifact);
            }
            None => {
                binaries.insert(
                    artifact_id.clone(),
                    download_artifact(
                        artifact_id.clone(),
                        tx_path_mutex.clone(),
                        rx_artifact_mutex.clone(),
                    ),
                );
            }
        },
        ArtifactHandler::NPYTabularArtifact { arrays, views } => {
            handle_add_npy(arrays, &artifact_id, tx_path_mutex, rx_artifact_mutex)
        }
    }
}

fn handle_add_npy(
    arrays: &mut HashMap<ArtifactId, NPYArray>,
    artifact_id: &ArtifactId,
    tx_path_mutex: &mut Arc<Mutex<Sender<i32>>>,
    rx_artifact_mutex: &Arc<Mutex<Receiver<ArtifactTransfer>>>,
) {
    match arrays.get_mut(artifact_id) {
        None => {
            arrays.insert(
                artifact_id.clone(),
                NPYArray::Loading(download_artifact(
                    artifact_id.clone(),
                    tx_path_mutex.clone(),
                    rx_artifact_mutex.clone(),
                )),
            );
        }
        Some(npyarray) => {
            let mut new_npyarray = None;
            match npyarray {
                NPYArray::Loading(binary_artifact) => {
                    poll_artifact_download(binary_artifact);
                    match binary_artifact {
                        BinaryArtifact::Loading(_) => {}
                        BinaryArtifact::Loaded(binary_data) => match load_npy_bytes(binary_data) {
                            Ok(nparray) => {
                                new_npyarray = Some(NPYArray::Loaded(nparray));
                            }
                            Err(err) => {
                                new_npyarray = Some(NPYArray::Error(err.to_string()));
                            }
                        },
                        BinaryArtifact::Error(err) => {
                            new_npyarray = Some(NPYArray::Error(err.to_string()));
                        }
                    }
                }
                _ => {}
            }
            if let Some(new_npyarray) = new_npyarray {
                *npyarray = new_npyarray;
            }
        }
    }
}

fn image_from_ndarray_healpix(
    array: &ndarray::prelude::ArrayBase<
        ndarray::OwnedRepr<f32>,
        ndarray::prelude::Dim<ndarray::IxDynImpl>,
    >,
    view: &NPYArtifactView,
) -> egui::ColorImage {
    let max = array.max().unwrap();
    let min = array.min().unwrap();
    let t = |x: f32| (x - min) / (max - min);
    let width = 1000;
    let height = 1000;
    let mut img = egui::ColorImage::new([width, height], egui::Color32::WHITE);
    let mut local_index = vec![0; array.shape().len()];
    for (dim_idx, dim) in view.index.iter().enumerate() {
        local_index[dim_idx] = *dim;
    }
    let ndim = local_index.len();
    for y in 0..height {
        for x in 0..width {
            // let lon_x = x as f64 / width as f64 * 8.0;
            // let lat_y = (y as f64 - height as f64 / 2.0) / (height as f64 / 2.0) * 2.0;
            // let (lon, lat) = cdshealpix::unproj(lon_x, lat_y);
            let (lon, lat) = (
                x as f64 / width as f64 * 2.0 * std::f64::consts::PI,
                -(y as f64 - height as f64 / 2.0) / (height as f64 / 2.0) * std::f64::consts::PI
                    / 2.0,
            );
            let nside = ((array.shape().last().unwrap() / 12) as f32).sqrt() as u32;
            let depth = cdshealpix::depth(nside);
            let hp_idx = cdshealpix::nested::hash(depth, lon, lat);
            // if hp_idx < cdshealpix::n_hash(depth) {
            // println!("nside {}, depth {}, idx {}", nside, depth, hp_idx);
            // dbg!(array.shape());
            local_index[ndim - 1] = hp_idx as usize;
            let color =
                CIVIDIS.eval_continuous(t(*array.get(local_index.as_slice()).unwrap()) as f64);
            img.pixels[y * width + x] = egui::Color32::from_rgb(color.r, color.g, color.b);
            // }
        }
    }
    img
}

struct GuiRuns {
    runs: Runs,
    dirty: bool,
    db_train_runs_sender: Sender<Vec<String>>,
    db_reciever: Receiver<HashMap<String, Run>>,
    recomputed_reciever: Receiver<HashMap<String, Run>>,
    dirty_sender: Sender<(GuiParams, HashMap<String, Run>)>,
    tx_db_artifact_path: Arc<Mutex<Sender<i32>>>,
    rx_db_artifact: Arc<Mutex<Receiver<ArtifactTransfer>>>,
    rx_batch_status: Receiver<(usize, usize)>,
    batch_status: (usize, usize),
    initialized: bool,
    data_status: DataStatus,
    gui_params: GuiParams,
    artifact_handlers: HashMap<String, ArtifactHandler>,
    args: Args, // texture: Option<egui::TextureHandle>,
}

fn recompute(runs: &mut HashMap<String, Run>, gui_params: &GuiParams) {
    resample(runs, gui_params);
}

fn get_train_ids_from_filter(runs: &HashMap<String, Run>, gui_params: &GuiParams) -> Vec<String> {
    if gui_params
        .param_filters
        .values()
        .map(|hs| hs.is_empty())
        .all(|x| x)
    {
        return Vec::new();
    }
    runs.iter()
        .filter_map(|run| {
            for (param_name, values) in gui_params
                .param_filters
                .iter()
                .filter(|(_, vs)| !vs.is_empty())
            {
                if let Some(run_value) = run.1.params.get(param_name) {
                    if !values.contains(run_value) {
                        return None;
                    }
                } else {
                    return None;
                }
                // if let Some(time_filter) = gui_params.time_filter {
                //     if run.1.created_at < time_filter {
                //         return None;
                //     }
                // }
            }
            Some(run.0)
        })
        .cloned()
        .collect()
}

fn resample(runs: &mut HashMap<String, Run>, gui_params: &GuiParams) {
    for run in runs.values_mut() {
        for metric in run.metrics.values_mut() {
            let max_n = gui_params.max_n;
            metric.values = if metric.orig_values.len() > max_n {
                let window = metric.orig_values.len() as i64 / (2 * max_n) as i64;
                let window = window.min(1);
                let didx = metric.orig_values.len() as f64 / max_n as f64;
                (0..max_n)
                    .map(|idx| {
                        let mean_value = (-window..=window)
                            .map(|sub_idx| {
                                metric.orig_values
                                    [((didx * idx as f64) as i64 + sub_idx).max(0) as usize][1]
                            })
                            .sum::<f64>()
                            / (-window..=window).count() as f64;
                        // let middle_x =
                        // metric.orig_values[((didx * idx as f64) as i64).max(0) as usize][0];
                        let middle_x = match gui_params.x_axis {
                            XAxis::Batch => {
                                metric.orig_values[((didx * idx as f64) as i64).max(0) as usize][0]
                            }
                            XAxis::Time => {
                                metric.orig_values[((didx * idx as f64) as i64).max(0) as usize][2]
                            }
                        };

                        [middle_x, mean_value]
                    })
                    .collect()
            } else {
                match gui_params.x_axis {
                    XAxis::Batch => metric.orig_values.iter().map(|x| [x[0], x[1]]).collect(),
                    XAxis::Time => metric.orig_values.iter().map(|x| [x[2], x[1]]).collect(),
                }
            };
            let fvalues: Vec<[f64; 2]> = (0..metric.values.len())
                .map(|orig_idx| {
                    let window = if gui_params.n_average > 0 {
                        -(gui_params.n_average.min(orig_idx) as i32)
                            ..=(gui_params.n_average.min(metric.values.len() - 1 - orig_idx) as i32)
                    } else {
                        0..=0
                    };
                    // let vals: Vec<f64> = window
                    //     .map(|sub_idx| {
                    //         let idx = orig_idx as i32 + sub_idx;
                    //         metric.values[idx as usize][1]
                    //     })
                    //     .sorted_by(|a, b| a.partial_cmp(b).unwrap())
                    //     .collect();
                    // let mean_val = vals[vals.len() / 2];
                    let sum: f64 = window
                        .clone()
                        .map(|sub_idx| {
                            let idx = orig_idx as i32 + sub_idx;
                            metric.values[idx as usize][1]
                        })
                        .sum();
                    let mean_val = sum / window.count() as f64;
                    [
                        metric.values[orig_idx][0],
                        mean_val,
                        // metric.values[orig_idx][2],
                    ]
                })
                .collect();
            metric.resampled = fvalues;
        }
    }
}

impl eframe::App for GuiRuns {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if !self.initialized {
            ctx.set_zoom_factor(2.0);
        }
        self.runs.active_runs = get_train_ids_from_filter(&self.runs.runs, &self.gui_params);
        self.runs.active_runs_time_ordered = self
            .runs
            .active_runs
            .iter()
            .cloned()
            .map(|train_id| {
                (
                    train_id.clone(),
                    self.runs.runs.get(&train_id).unwrap().created_at,
                )
            })
            .sorted_by_key(|(_train_id, created_at)| *created_at)
            .collect();
        self.runs.time_filtered_runs = self
            .runs
            .active_runs_time_ordered
            .iter()
            .cloned()
            .filter(|(_train_id, created_at)| {
                if let Some(time_filter) = self.gui_params.time_filter {
                    *created_at >= time_filter
                } else {
                    true
                }
            })
            .map(|(train_id, _)| train_id)
            .collect();
        if self.dirty {
            self.dirty_sender
                .send((self.gui_params.clone(), self.runs.runs.clone()))
                .expect("Failed to send dirty runs");
            self.db_train_runs_sender
                .send(self.runs.active_runs.clone())
                .expect("Failed to send train runs to db thread");
            self.dirty = false;
            // self.recompute();
        }
        if let Ok(new_runs) = self.recomputed_reciever.try_recv() {
            // println!("[app] recieved recomputed runs");
            self.runs.runs = new_runs;
            if self.data_status == DataStatus::FirstDataArrived && self.runs.runs.len() > 0 {
                self.data_status = DataStatus::FirstDataProcessed;
            }
        }
        if let Ok(new_runs) = self.db_reciever.try_recv() {
            for train_id in new_runs.keys() {
                if !self.runs.runs.contains_key(train_id) {
                    let new_active = get_train_ids_from_filter(&new_runs, &self.gui_params);
                    println!("[app] recieved new runs, sending to compute...");
                    self.db_train_runs_sender
                        .send(new_active)
                        .expect("Failed to send train runs to db thread");
                    break;
                }
            }
            self.dirty_sender
                .send((self.gui_params.clone(), new_runs))
                .expect("Failed to send dirty runs");
            if self.data_status == DataStatus::Waiting {
                self.data_status = DataStatus::FirstDataArrived;
            }
            // self.recompute();
        }

        let ensemble_colors: HashMap<String, egui::Color32> = self
            .runs
            .runs
            .values()
            .map(|run| label_from_active_inspect_params(run, &self.gui_params))
            .unique()
            .sorted()
            .enumerate()
            .map(|(idx, ensemble_id)| {
                let h = idx as f32 * 0.61;
                let color: egui::Color32 = epaint::Hsva::new(h, 0.85, 0.5, 1.0).into();
                (ensemble_id.clone(), color)
            })
            .collect();
        let run_ensemble_color: HashMap<String, egui::Color32> = self
            .runs
            .runs
            .values()
            .map(|run| {
                let train_id = run.params.get("train_id").unwrap();
                let ensemble_id = label_from_active_inspect_params(run, &self.gui_params); // run.params.get("ensemble_id").unwrap();
                (
                    train_id.clone(),
                    *ensemble_colors.get(&ensemble_id).unwrap(),
                )
            })
            .collect();
        let param_values = get_parameter_values(&self.runs, true);
        for param_name in param_values.keys() {
            if !self.gui_params.param_filters.contains_key(param_name) {
                self.gui_params
                    .param_filters
                    .insert(param_name.clone(), HashSet::new());
            }
        }
        // for run in &filtered_runs {
        //     println!("{:?}", run.1.params.get("epochs"))
        // }
        let filtered_values = get_parameter_values(&self.runs, false);
        let metric_names: Vec<String> = self
            .runs
            .time_filtered_runs
            .iter()
            .map(|train_id| self.runs.runs.get(train_id).unwrap())
            .map(|run| run.metrics.keys().cloned())
            .flatten()
            .unique()
            .sorted()
            .collect();

        egui::SidePanel::left("Controls")
            .resizable(true)
            .default_width(200.0)
            .min_width(200.0)
            // .width_range(100.0..=600.0)
            .show(ctx, |ui| {
                self.render_parameters(ui, param_values, filtered_values, ctx);
            });
        egui::SidePanel::right("Metrics")
            .resizable(true)
            .default_width(300.0)
            .width_range(100.0..=500.0)
            .show(ctx, |ui| {
                self.render_metrics(ui, &metric_names);
                ui.separator();
                self.render_artifact_selector(ui);
            });
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    if ui.button("Time").clicked() {
                        self.gui_params.x_axis = XAxis::Time;
                    }
                    if ui.button("Batch").clicked() {
                        self.gui_params.x_axis = XAxis::Batch;
                    }
                    self.render_time_selector(ui);
                    if let Ok(batch_status) = self.rx_batch_status.try_recv() {
                        self.batch_status = batch_status;
                    }
                });
                if self.batch_status.1 > 0 {
                    ui.add(egui::ProgressBar::new(
                        self.batch_status.0 as f32 / self.batch_status.1 as f32,
                    ));
                }
            });

            ui.separator();
            egui::ScrollArea::vertical()
                .id_source("central_space")
                .show(ui, |ui| {
                    self.render_artifacts(ui, &run_ensemble_color);
                    self.render_plots(ui, metric_names, run_ensemble_color);
                });
        });
        self.initialized = true;
        ctx.request_repaint();
    }
}

fn get_artifact_type(path: &String) -> &str {
    path.split(".").last().unwrap_or("unknown")
}

fn label_from_active_inspect_params(run: &Run, gui_params: &GuiParams) -> String {
    let label = if gui_params.inspect_params.is_empty() {
        run.params.get("ensemble_id").unwrap().clone()
    } else {
        let empty = "".to_string();
        gui_params
            .inspect_params
            .iter()
            .sorted()
            .map(|param| {
                format!(
                    "{}:{}",
                    param.split(".").last().unwrap_or(param),
                    run.params.get(param).unwrap_or(&empty)
                )
            })
            .join(", ")
    };
    label
}
fn show_artifacts(
    ui: &mut egui::Ui,
    handler: &mut ArtifactHandler,
    gui_params: &GuiParams,
    runs: &HashMap<String, Run>,
    filtered_runs: &Vec<String>,
    run_ensemble_color: &HashMap<String, egui::Color32>,
) {
    match handler {
        ArtifactHandler::NPYHealPixArtifact {
            arrays,
            textures,
            views,
        } => {
            // let texture = texture.get_or_insert_with(|| {});
            let npy_axis_id = ui.id().with("npy_axis");
            let available_artifact_names: Vec<&String> = arrays.keys().map(|id| &id.name).collect();
            for (artifact_name, filtered_arrays) in gui_params
                .artifact_filters
                .iter()
                .filter(|name| available_artifact_names.contains(name))
                .map(|name| {
                    (
                        name,
                        arrays.iter().filter(|(key, _v)| {
                            key.name == *name && filtered_runs.contains(&key.train_id)
                        }),
                    )
                })
            {
                // artifact_name,
                let plot_width = ui.available_width() * 0.48;
                let plot_height = ui.available_width() * 0.48 * 0.5;
                ui.horizontal_wrapped(|ui| {
                    for (artifact_name, array_group) in
                        &filtered_arrays.group_by(|(aid, _)| aid.name.clone())
                    {
                        ui.group(|ui| {
                            ui.label(egui::RichText::new(artifact_name).size(20.0));
                            ui.allocate_space(egui::Vec2::new(ui.available_width(), 0.0));
                            for (artifact_id, array) in array_group {
                                match array {
                                    NPYArray::Loading(binary_artifact) => {
                                        render_artifact_download_progress(binary_artifact, ui);
                                    }
                                    NPYArray::Loaded(array) => {
                                        // ui.allocate_ui()
                                        ui.allocate_ui(
                                            egui::Vec2::from([plot_width, plot_height + 200.0]),
                                            |ui| {
                                                render_npy_artifact(
                                                    ui,
                                                    runs,
                                                    artifact_id,
                                                    gui_params,
                                                    run_ensemble_color,
                                                    views,
                                                    array,
                                                    textures,
                                                    plot_width,
                                                    npy_axis_id,
                                                );
                                            },
                                        );
                                    }
                                    NPYArray::Error(err) => {
                                        // ui.colored_label(egui::Color32::RED, err);
                                        // ui.allocate_space(egui::Vec2::new(ui.available_width(), 0.0));
                                    }
                                }
                            }
                        });
                    }
                });
            }
        }
        ArtifactHandler::ImageArtifact { binaries } => {
            let max_size = egui::Vec2::new(ui.available_width() / 2.1, ui.available_height() / 2.1);
            let available_artifact_names: Vec<&String> =
                binaries.keys().map(|id| &id.name).collect();
            ui.horizontal_wrapped(|ui| {
                // ui.set_max_width(ui.available_width());
                // ui.allocate_space(egui::Vec2::new(ui.available_width(), 0.0));
                for (artifact_name, filtered_arrays) in gui_params
                    .artifact_filters
                    .iter()
                    .filter(|name| available_artifact_names.contains(name))
                    .map(|name| {
                        (
                            name,
                            binaries.iter().filter(|(key, _v)| {
                                key.name == *name && filtered_runs.contains(&key.train_id)
                            }),
                        )
                    })
                {
                    for (artifact_id, binary_artifact) in filtered_arrays {
                        ui.allocate_ui(max_size, |ui| {
                            ui.push_id(artifact_id, |ui| {
                                ui.vertical(|ui| {
                                    let label = label_from_active_inspect_params(
                                        runs.get(&artifact_id.train_id).unwrap(),
                                        &gui_params,
                                    );
                                    ui.colored_label(
                                        run_ensemble_color
                                            .get(&artifact_id.train_id)
                                            .unwrap()
                                            .clone(),
                                        format!("{}: {}", artifact_id.name, label),
                                    );
                                    // ui.allocate_space(max_size);
                                    // if let Some(binary) = binaries.get(&artifact_id) {
                                    match binary_artifact {
                                        BinaryArtifact::Loaded(binary_data) => {
                                            ui.add(
                                                egui::Image::from_bytes(
                                                    format!(
                                                        "bytes://{}:{}",
                                                        artifact_id.train_id, artifact_id.name
                                                    ),
                                                    binary_data.clone(),
                                                )
                                                .fit_to_exact_size(max_size),
                                            );
                                        }
                                        BinaryArtifact::Loading((_, status)) => {
                                            if status.status.size > 0 {
                                                ui.label(format!(
                                                    "{:.1}/{:.1}",
                                                    status.status.downloaded as f32 / 1e6,
                                                    status.status.size as f32 / 1e6
                                                ));
                                                ui.add(egui::ProgressBar::new(
                                                    status.status.downloaded as f32
                                                        / status.status.size as f32,
                                                ));
                                            }
                                        }
                                        BinaryArtifact::Error(err) => {
                                            ui.label(err);
                                        }
                                    }
                                });
                            });
                        });
                    }
                }
            });
        }
        ArtifactHandler::NPYTabularArtifact { arrays, views } => {
            let plot_width = ui.available_width() * 0.48;
            let plot_height = ui.available_width() * 0.48 * 0.5;
            let available_artifact_names: Vec<&String> = arrays.keys().map(|id| &id.name).collect();
            // println!("render tabular");
            for (artifact_name, filtered_arrays) in gui_params
                .artifact_filters
                .iter()
                .filter(|name| available_artifact_names.contains(name))
                .map(|name| {
                    (
                        name,
                        arrays.iter().filter(|(key, _v)| {
                            key.name == *name && filtered_runs.contains(&key.train_id)
                        }),
                    )
                })
            {
                // println!("looper");
                for (artifact_id, array) in filtered_arrays {
                    match array {
                        NPYArray::Loading(binary_artifact) => {
                            render_artifact_download_progress(binary_artifact, ui);
                            // println!("loading");
                        }
                        NPYArray::Loaded(array) => {
                            // println!("loaded");
                            // ui.allocate_ui()
                            ui.allocate_ui(
                                egui::Vec2::from([plot_width, plot_height + 200.0]),
                                |ui| {
                                    render_npy_artifact_tabular(
                                        ui,
                                        runs,
                                        artifact_id,
                                        gui_params,
                                        run_ensemble_color,
                                        views,
                                        array,
                                        // textures,
                                        plot_width,
                                        // npy_axis_id,
                                    );
                                },
                            );
                        }
                        NPYArray::Error(err) => {
                            // println!("error");
                            ui.label(err);
                            // ui.colored_label(egui::Color32::RED, err);
                            // ui.allocate_space(egui::Vec2::new(ui.available_width(), 0.0));
                        }
                    }
                }
            }
        }
    }
}

fn render_artifact_download_progress(binary_artifact: &BinaryArtifact, ui: &mut egui::Ui) {
    match binary_artifact {
        BinaryArtifact::Loading((_, status)) => {
            ui.label("waiting for first sync...");
            if status.status.size > 0 {
                ui.label(format!(
                    "{:.1}/{:.1}",
                    status.status.downloaded as f32 / 1e6,
                    status.status.size as f32 / 1e6
                ));
                ui.add(egui::ProgressBar::new(
                    status.status.downloaded as f32 / status.status.size as f32,
                ));
            }
        }
        BinaryArtifact::Loaded(data) => {
            ui.label("loaded!");
        }
        BinaryArtifact::Error(err) => {
            ui.label(err);
        }
    }
}

fn render_npy_artifact(
    ui: &mut egui::Ui,
    runs: &HashMap<String, Run>,
    artifact_id: &ArtifactId,
    gui_params: &GuiParams,
    run_ensemble_color: &HashMap<String, egui::Color32>,
    views: &mut HashMap<ArtifactId, NPYArtifactView>,
    array: &ndarray::prelude::ArrayBase<
        ndarray::OwnedRepr<f32>,
        ndarray::prelude::Dim<ndarray::IxDynImpl>,
    >,
    textures: &mut HashMap<NPYArtifactView, egui::TextureHandle>,
    plot_width: f32,
    npy_axis_id: egui::Id,
) {
    ui.vertical(|ui| {
        let label =
            label_from_active_inspect_params(runs.get(&artifact_id.train_id).unwrap(), &gui_params);
        ui.colored_label(
            run_ensemble_color
                .get(&artifact_id.train_id)
                .unwrap()
                .clone(),
            format!("{}: {}", artifact_id.name, label),
        );
        let view = views.entry(artifact_id.clone()).or_insert(NPYArtifactView {
            artifact_id: artifact_id.clone(),
            index: vec![0; array.shape().len() - 1],
        });
        for (dim_idx, dim) in array
            .shape()
            .iter()
            .enumerate()
            .take(array.shape().len() - 1)
        {
            ui.add(egui::Slider::new(&mut view.index[dim_idx], 0..=(dim - 1)));
        }
        ui.label(array.shape().iter().map(|x| x.to_string()).join(","));
        if !textures.contains_key(&view) {
            let mut texture = ui.ctx().load_texture(
                &artifact_id.name,
                egui::ColorImage::example(),
                egui::TextureOptions::default(),
            );
            let img = image_from_ndarray_healpix(array, view);
            texture.set(img, egui::TextureOptions::default());
            textures.insert(view.clone(), texture);
        }
        let pi = egui_plot::PlotImage::new(
            textures.get(&view).unwrap(),
            // texture.id(),
            egui_plot::PlotPoint::from([0.0, 0.0]),
            [2.0 * 3.14, 3.14],
        );
        // texture.set(img, egui::TextureOptions::default());
        // ui.image((texture.id(), texture.size_vec2()));
        Plot::new(artifact_id)
            .width(plot_width)
            .height(plot_width / 2.0)
            .data_aspect(1.0)
            .view_aspect(1.0)
            .show_grid(false)
            .link_axis(npy_axis_id, true, true)
            .link_cursor(npy_axis_id, true, true)
            .show(ui, |plot_ui| {
                plot_ui.image(pi);
            });
    });
}

fn render_npy_artifact_tabular(
    ui: &mut egui::Ui,
    runs: &HashMap<String, Run>,
    artifact_id: &ArtifactId,
    gui_params: &GuiParams,
    run_ensemble_color: &HashMap<String, egui::Color32>,
    views: &mut HashMap<ArtifactId, NPYTabularArtifactView>,
    array: &ndarray::prelude::ArrayBase<
        ndarray::OwnedRepr<f32>,
        ndarray::prelude::Dim<ndarray::IxDynImpl>,
    >,
    plot_width: f32,
) {
    ui.vertical(|ui| {
        let label =
            label_from_active_inspect_params(runs.get(&artifact_id.train_id).unwrap(), &gui_params);
        ui.colored_label(
            run_ensemble_color
                .get(&artifact_id.train_id)
                .unwrap()
                .clone(),
            format!("{}: {}", artifact_id.name, label),
        );
        let view = views
            .entry(artifact_id.clone())
            .or_insert(NPYTabularArtifactView {
                artifact_id: artifact_id.clone(),
                x: 0,
                y: 0,
                c: 0,
                log_x: false,
                log_y: false, // index: vec![0; array.shape().len() - 1],
            });
        // for (dim_idx, dim) in array
        //     .shape()
        //     .iter()
        //     .enumerate()
        //     .take(array.shape().len() - 1)
        // {
        let n_rows = array.shape()[0];
        let n_cols = array.shape()[1];
        ui.add(egui::Slider::new(&mut view.x, 0..=(n_cols - 1)));
        ui.add(egui::Slider::new(&mut view.y, 0..=(n_cols - 1)));
        ui.add(egui::Slider::new(&mut view.c, 0..=(n_cols - 1)));
        ui.checkbox(&mut view.log_x, "log x-axis");
        ui.checkbox(&mut view.log_y, "log x-axis");
        // }
        // ui.label(array.shape().iter().map(|x| x.to_string()).join(","));
        // texture.set(img, egui::TextureOptions::default());
        // ui.image((texture.id(), texture.size_vec2()));
        let plot = Plot::new(artifact_id)
            .width(plot_width)
            .height(plot_width / 2.0)
            .auto_bounds_x()
            .auto_bounds_y()
            // .data_aspect(1.0)
            // .view_aspect(1.0)
            .show_grid(true)
            // .link_axis(npy_axis_id, true, true)
            // .link_cursor(npy_axis_id, true, true)
            .show(ui, |plot_ui| {
                let xs = array.slice(s![.., view.x]);
                let ys = array.slice(s![.., view.y]);
                let xy = ndarray::concatenate![
                    ndarray::Axis(1),
                    xs.insert_axis(ndarray::Axis(1)),
                    ys.insert_axis(ndarray::Axis(1))
                ];
                // xs.view()
                // println!("{:?}", xy.shape());
                let x_shaper = |x: f32| {
                    if view.log_x {
                        if x > 0.000001 {
                            x.log10()
                        } else {
                            0.0
                        }
                    } else {
                        x
                    }
                };
                let y_shaper = |y: f32| {
                    if view.log_y {
                        if y > 0.000001 {
                            y.log10()
                        } else {
                            0.0
                        }
                    } else {
                        y
                    }
                };
                let xy = xy
                    .outer_iter()
                    .map(|row| {
                        [
                            x_shaper(*row.get(0).unwrap()) as f64,
                            y_shaper(*row.get(1).unwrap()) as f64,
                        ]
                    })
                    .collect_vec();
                // let ys = array.slice_collapse(s![.., view.y]);
                // let xy = ndarray::stack(ndarray::Axis(1), &[xs, ys]).unwrap();
                // println!("{:?}", xy.shape());
                // let xy = xs
                //     .to_slice()
                //     .unwrap()
                //     .iter()
                //     .zip(ys.to_slice().unwrap())
                //     .map(|x| [*x.0 as f64, *x.1 as f64])
                //     .collect_vec();
                // println!("{:?}", v);
                // let v = xy.iter().next();
                // xs.fold_axis(, , )
                plot_ui.points(Points::new(xy));
                // plot_ui.image(pi);
            });
    });
}

impl GuiRuns {
    fn render_parameters(
        &mut self,
        ui: &mut egui::Ui,
        param_values: HashMap<String, HashSet<String>>,
        filtered_values: HashMap<String, HashSet<String>>,
        ctx: &egui::Context,
    ) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.vertical(|ui| {
                if ui
                    .add(egui::TextEdit::singleline(
                        &mut self.gui_params.param_name_filter,
                    ))
                    .changed()
                {}
                if ui
                    .add(
                        egui::Slider::new(&mut self.gui_params.n_average, 0usize..=200usize)
                            .logarithmic(true),
                    )
                    .changed()
                {
                    self.dirty = true;
                };
                if ui
                    .add(
                        egui::Slider::new(&mut self.gui_params.max_n, 500usize..=2000usize)
                            .logarithmic(true),
                    )
                    .changed()
                {
                    self.dirty = true;
                };
                let param_names = param_values.keys().cloned().collect_vec();
                for name in param_names.iter().sorted() {
                    if let Some(values) = self.gui_params.param_filters.get(name) {
                        if !values.is_empty() {
                            egui::CollapsingHeader::new(name)
                                // .default_open(self.gui_params.param_name_filter.len() > 1)
                                .default_open(!name.ends_with("_id"))
                                .show(ui, |ui| {
                                    self.render_parameter_key(
                                        name,
                                        ui,
                                        &param_values,
                                        &filtered_values,
                                        ctx,
                                    );
                                });
                        }
                    }
                }
                ui.separator();
                self.render_parameters_one_level(
                    &param_values,
                    param_names,
                    ui,
                    &filtered_values,
                    ctx,
                    0,
                );
            });
        });
    }

    fn render_parameters_one_level(
        &mut self,
        param_values: &HashMap<String, HashSet<String>>,
        param_names: Vec<String>,
        ui: &mut egui::Ui,
        filtered_values: &HashMap<String, HashSet<String>>,
        ctx: &egui::Context,
        depth: usize,
    ) {
        let groups = param_names //param_values
            .iter()
            // .keys()
            .sorted()
            .group_by(|param_name| {
                param_name.split(".").take(depth + 1).join(".")
                // .unwrap_or(param_name)
            });
        for (param_group_name, param_group) in &groups {
            let param_group_vec = param_group
                .cloned()
                .map(|name| {
                    // if name.contains(".") {
                    // name.split_once(".").unwrap().1.to_string()
                    // name.replacen(".", "#", 1)
                    // } else {
                    name
                    // }
                })
                .collect_vec();
            let param_filter = self.gui_params.param_name_filter.as_str();
            if param_group_vec.iter().all(|name| {
                !param_values
                    .get(name)
                    .unwrap()
                    .iter()
                    .any(|value_str| value_str.to_lowercase().contains(param_filter))
                    && !name.contains(param_filter)
            }) {
                continue;
            }
            // ui.collapsing(param_group_name, |ui| {
            egui::CollapsingHeader::new(&param_group_name)
                // .default_open(self.gui_params.param_name_filter.len() > 1)
                .default_open(!param_group_name.ends_with("_id"))
                .show(ui, |ui| {
                    if param_group_vec
                        .iter()
                        .any(|name| name.split(".").count() > depth + 1)
                    {
                        // println!(
                        //     "{:?}: {}",
                        //     param_group_vec, self.gui_params.param_name_filter
                        // );
                        self.render_parameters_one_level(
                            param_values,
                            param_group_vec,
                            ui,
                            filtered_values,
                            ctx,
                            depth + 1,
                        );
                    } else {
                        for param_name in &param_group_vec {
                            if !param_values
                                .get(param_name)
                                .unwrap()
                                .iter()
                                .any(|value_str| {
                                    value_str
                                        .to_lowercase()
                                        .contains(self.gui_params.param_name_filter.as_str())
                                })
                                && !param_name.contains(self.gui_params.param_name_filter.as_str())
                            {
                                continue;
                            }
                            // ui.
                            // ui.separator();
                            self.render_parameter_key(
                                param_name,
                                ui,
                                param_values,
                                filtered_values,
                                ctx,
                            );
                        }
                    }
                });
        }
    }

    fn render_parameter_key(
        &mut self,
        param_name: &String,
        ui: &mut egui::Ui,
        param_values: &HashMap<String, HashSet<String>>,
        filtered_values: &HashMap<String, HashSet<String>>,
        ctx: &egui::Context,
    ) {
        let frame_border = if self.gui_params.inspect_params.contains(param_name) {
            1.0
        } else {
            0.0
        };
        // ui.allocate_ui(egui::Vec2::new(0.0, 0.0), |ui| {})
        let param_frame = egui::Frame::none()
            // .fill(egui::Color32::GREEN)
            .stroke(egui::Stroke::new(frame_border, egui::Color32::GREEN))
            .show(ui, |ui| {
                ui.allocate_space(egui::Vec2::new(ui.available_width(), 0.0));
                // ui.label(param_name);
                ui.horizontal_wrapped(|ui| {
                    self.render_parameter_values(
                        &param_values,
                        &param_name,
                        &filtered_values,
                        ctx,
                        ui,
                    );
                });
            });
        // println!("{:?}", param_frame.response.sense);
        if param_frame
            .response
            .interact(egui::Sense::click())
            .clicked()
        {
            if self.gui_params.inspect_params.contains(param_name) {
                self.gui_params.inspect_params.remove(param_name);
            } else {
                self.gui_params
                    .inspect_params
                    .insert(param_name.to_string());
            }
        }
    }

    fn render_parameter_values(
        &mut self,
        param_values: &HashMap<String, HashSet<String>>,
        param_name: &str,
        filtered_values: &HashMap<String, HashSet<String>>,
        ctx: &egui::Context,
        ui: &mut egui::Ui,
    ) {
        let param_name = param_name.replace("#", ".");
        for value in param_values
            .get(&param_name)
            .unwrap()
            .iter()
            .sorted_by(|name1, name2| {
                if let Ok(float1) = name1.parse::<f32>() {
                    if let Ok(float2) = name2.parse::<f32>() {
                        let order = float1.partial_cmp(&float2).unwrap();
                        if order != core::cmp::Ordering::Equal {
                            return order;
                        }
                        return name1.cmp(name2);
                        // let s1 = name1.parse::<String>().unwrap();
                        // let s2 = name1.parse::<String>().unwrap();
                    }
                }
                name1.cmp(name2)
            })
        {
            let active_filter = self
                .gui_params
                .param_filters
                .get(&param_name)
                .unwrap()
                .contains(value);
            let filtered_runs_contains = if let Some(values) = filtered_values.get(&param_name) {
                values.contains(value)
            } else {
                false
            };
            let color = if filtered_runs_contains {
                egui::Color32::LIGHT_GREEN
            } else {
                ctx.style().visuals.widgets.inactive.bg_fill
            };
            let button = ui.add(
                egui::Button::new(value)
                    .stroke(egui::Stroke::new(1.0, color))
                    .selected(active_filter),
            );
            if button.clicked_by(egui::PointerButton::Middle) {
                ui.output_mut(|output| {
                    output.copied_text = value.clone();
                });
            } else if button.clicked() {
                if self
                    .gui_params
                    .param_filters
                    .get(&param_name)
                    .unwrap()
                    .contains(value)
                {
                    self.gui_params
                        .param_filters
                        .get_mut(&param_name)
                        .unwrap()
                        .remove(value);
                } else {
                    self.gui_params
                        .param_filters
                        .get_mut(&param_name)
                        .unwrap()
                        .insert(value.clone());
                    dbg!(&param_name, value);
                }
                self.dirty = true;
            }
        }
    }

    fn render_plots(
        &mut self,
        ui: &mut egui::Ui,
        metric_names: Vec<String>,
        run_ensemble_color: HashMap<String, egui::Color32>,
    ) {
        let xaxis_ids: HashMap<_, _> = self
            .runs
            .runs
            .values()
            .map(|run| run.metrics.values().map(|metric| metric.xaxis.clone()))
            .flatten()
            .unique()
            .sorted()
            .map(|xaxis| (xaxis.clone(), ui.id().with(xaxis)))
            .collect();

        let metric_name_axis_id: HashMap<_, _> = self
            .runs
            .runs
            .values()
            .map(|run| {
                run.metrics.iter().map(|(metric_name, metric)| {
                    (metric_name, xaxis_ids.get(&metric.xaxis).unwrap())
                })
            })
            .flatten()
            .unique()
            .collect();

        // let link_group_id = ui.id().with("linked_demo");
        let filtered_metric_names: Vec<String> = metric_names
            .into_iter()
            .filter(|name| {
                self.gui_params.metric_filters.contains(name)
                    || self.gui_params.metric_filters.is_empty()
            })
            .collect();
        let plot_width = if filtered_metric_names.len() <= 2 {
            ui.available_width() / 2.1
        } else {
            ui.available_width() / 2.1
        };
        let plot_height = if filtered_metric_names.len() <= 2 {
            ui.available_width() / 4.1
        } else {
            ui.available_width() / 4.1
        };
        let x_axis = self.gui_params.x_axis.clone();
        let formatter = match x_axis {
            XAxis::Time => |name: &str, value: &PlotPoint| {
                let ts = NaiveDateTime::from_timestamp_opt(value.x as i64, 0).unwrap();
                let xstr = ts.format("%y/%m/%d - %H:%M").to_string();
                format!("time: {}\ny: {}", xstr, value.y)
            },
            XAxis::Batch => {
                |name: &str, value: &PlotPoint| format!("x:{:.3}\ny:{:.3}", value.x, value.y)
            }
        };
        let plots: HashMap<_, _> = filtered_metric_names
            .into_iter()
            .map(|metric_name| {
                (
                    metric_name.clone(),
                    Plot::new(&metric_name)
                        .auto_bounds_x()
                        .auto_bounds_y()
                        .legend(Legend::default())
                        .width(plot_width)
                        .height(plot_height)
                        // .label_formatter(match x_axis {
                        //     XAxis::Time => |name, value: &PlotPoint| {
                        //         let ts =
                        //             NaiveDateTime::from_timestamp_opt(value.x as i64, 0).unwrap();
                        //         let xstr = ts.format("%y/%m/%d-%Hh-%Mm").to_string();
                        //         format!("time: {}\ny: {}", xstr, value.y)
                        //     },
                        //     XAxis::Batch => |name, value: &PlotPoint| {
                        //         format!("x:{:.3}\ny:{:.3}", value.x, value.y)
                        //     },
                        // })
                        .label_formatter(formatter)
                        .x_axis_formatter(match x_axis {
                            XAxis::Time => |value, n_chars, range: &RangeInclusive<f64>| {
                                let ts =
                                    NaiveDateTime::from_timestamp_opt(value as i64, 0).unwrap();
                                let delta = range.end() - range.start();
                                if delta > (5 * 24 * 60 * 60) as f64 {
                                    ts.format("%m/%d").to_string()
                                } else if delta > (5 * 60 * 60) as f64 {
                                    ts.format("%d-%Hh").to_string()
                                } else {
                                    ts.format("%Hh:%Mm").to_string()
                                }
                            },
                            XAxis::Batch => |value, n_chars, range: &RangeInclusive<f64>| {
                                format!("{}", value as i64).to_string()
                            },
                        })
                        .link_axis(
                            **metric_name_axis_id.get(&metric_name).unwrap(),
                            true,
                            false,
                        )
                        .link_cursor(**metric_name_axis_id.get(&metric_name).unwrap(), true, true),
                )
            })
            .collect();
        // egui::ScrollArea::vertical().show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            for (metric_name, plot) in plots.into_iter().sorted_by_key(|(k, _v)| k.clone()) {
                ui.allocate_ui(egui::Vec2::from([plot_width, plot_height]), |ui| {
                    ui.vertical_centered(|ui| {
                        ui.label(&metric_name);
                        plot.show(ui, |plot_ui| {
                            if self.gui_params.n_average > 1 {
                                for (run_id, run) in self
                                    .runs
                                    .time_filtered_runs
                                    .iter()
                                    .sorted()
                                    .map(|train_id| {
                                        (train_id, self.runs.runs.get(train_id).unwrap())
                                    })
                                {
                                    if let Some(metric) = run.metrics.get(&metric_name) {
                                        // let label = self.label_from_active_inspect_params(run);
                                        plot_ui.line(
                                            Line::new(PlotPoints::from(metric.values.clone()))
                                                // .name(&label)
                                                .stroke(Stroke::new(
                                                    1.0,
                                                    run_ensemble_color
                                                        .get(run_id)
                                                        .unwrap()
                                                        .gamma_multiply(0.4),
                                                )),
                                        );
                                    }
                                }
                            }
                            for (run_id, run) in self
                                .runs
                                .time_filtered_runs
                                .iter()
                                .sorted()
                                .map(|train_id| (train_id, self.runs.runs.get(train_id).unwrap()))
                            {
                                if let Some(metric) = run.metrics.get(&metric_name) {
                                    let label =
                                        label_from_active_inspect_params(run, &self.gui_params);
                                    let running = metric.last_value
                                        > chrono::Utc::now()
                                            .naive_utc()
                                            .checked_sub_signed(chrono::Duration::minutes(5))
                                            .unwrap();
                                    let label = if running {
                                        format!("{} (r)", label)
                                    } else {
                                        label
                                    };
                                    plot_ui.line(
                                        Line::new(PlotPoints::from(metric.resampled.clone()))
                                            .name(&label)
                                            .stroke(Stroke::new(
                                                2.0 * if running { 2.0 } else { 1.0 },
                                                *run_ensemble_color.get(run_id).unwrap(),
                                            )),
                                    );
                                }
                            }
                        })
                    });
                    // });
                });
            }
        });
        // });
        if self.data_status == DataStatus::FirstDataProcessed {
            // plot_ui.set_auto_bounds(egui::Vec2b::new(true, true));
            self.data_status = DataStatus::FirstDataPlotted;
        }
    }

    fn render_metrics(&mut self, ui: &mut egui::Ui, metric_names: &Vec<String>) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.vertical(|ui| {
                for metric_name in metric_names {
                    let active_filter = self.gui_params.metric_filters.contains(metric_name);
                    if ui
                        .add(egui::Button::new(metric_name).selected(active_filter))
                        .clicked()
                    {
                        if self.gui_params.metric_filters.contains(metric_name) {
                            self.gui_params.metric_filters.remove(metric_name);
                        } else {
                            self.gui_params.metric_filters.insert(metric_name.clone());
                        }
                        self.dirty = true;
                    }
                }
            });
        });
    }

    fn render_artifact_selector(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical()
            .id_source("artifact_selector")
            .show(ui, |ui| {
                ui.allocate_space(egui::Vec2::new(ui.available_width(), 0.0));
                for artifact_name in self
                    .runs
                    .time_filtered_runs
                    // .active_runs
                    .iter()
                    .map(|train_id| self.runs.runs.get(train_id).unwrap().artifacts.keys())
                    .flatten()
                    .unique()
                    .sorted()
                {
                    if ui
                        .add(
                            egui::Button::new(artifact_name)
                                .selected(self.gui_params.artifact_filters.contains(artifact_name)),
                        )
                        .clicked()
                    {
                        if self.gui_params.artifact_filters.contains(artifact_name) {
                            self.gui_params.artifact_filters.remove(artifact_name);
                        } else {
                            self.gui_params
                                .artifact_filters
                                .insert(artifact_name.clone());
                        }
                    }
                }
            });
    }

    fn render_artifacts(
        &mut self,
        ui: &mut egui::Ui,
        run_ensemble_color: &HashMap<String, egui::Color32>,
    ) {
        let active_artifact_types: Vec<&str> = self
            .gui_params
            .artifact_filters
            .iter()
            .map(|artifact_name| {
                self.runs
                    .time_filtered_runs
                    .iter()
                    .map(|train_id| (train_id, self.runs.runs.get(train_id).unwrap()))
                    .map(|(_train_id, run)| {
                        if let Some(artifact_id) = run.artifacts.get(artifact_name) {
                            let artifact_type_str = get_artifact_type(&artifact_id.name);
                            if self.artifact_handlers.contains_key(artifact_type_str) {
                                // println!("{}", artifact_type_str);
                                // add_artifact(handler, ui, train_id, path);
                                artifact_type_str
                            } else {
                                ""
                            }
                            // if let Some(artifact_handler) = self.
                        } else {
                            ""
                        }
                    })
            })
            .flatten()
            .unique()
            .sorted()
            .collect();
        for artifact_type in active_artifact_types {
            if let Some(handler) = self.artifact_handlers.get_mut(artifact_type) {
                for (run_id, run) in self
                    .runs
                    .time_filtered_runs
                    .iter()
                    .map(|run_id| (run_id, self.runs.runs.get(run_id).unwrap()))
                {
                    for (artifact_name, artifact_id) in
                        run.artifacts.iter().filter(|&(art_name, _)| {
                            self.gui_params.artifact_filters.contains(art_name)
                        })
                    {
                        // println!("[artifacts] filtered {} {}", run_id, artifact_name);
                        if artifact_type == get_artifact_type(&artifact_id.name) {
                            // println!("{}", artifact_type);
                            add_artifact(
                                handler,
                                artifact_id,
                                &self.args,
                                &mut self.tx_db_artifact_path,
                                &self.rx_db_artifact,
                            );
                        }
                    }
                }
                show_artifacts(
                    ui,
                    handler,
                    &self.gui_params,
                    &self.runs.runs,
                    &self.runs.time_filtered_runs,
                    &run_ensemble_color,
                );
            }
            // if let Some(handler) = self.artifact_handlers.get(artifact_type) {}
        }
    }

    fn render_time_selector(&mut self, ui: &mut egui::Ui) {
        if self.runs.active_runs.len() > 1 {
            ui.group(|ui| {
                ui.spacing_mut().slider_width = ui.available_width() - 300.0;
                ui.label("Cut-off time");
                let time_slider = egui::Slider::new(
                    &mut self.gui_params.time_filter_idx,
                    0..=self.runs.active_runs.len() - 1,
                )
                .custom_formatter(|fval, _| {
                    let idx = fval as usize;
                    let created_at = self.runs.active_runs_time_ordered[idx].1;
                    created_at.to_string()
                });
                if ui.add(time_slider).changed() {
                    self.gui_params.time_filter =
                        Some(self.runs.active_runs_time_ordered[self.gui_params.time_filter_idx].1)
                }
            });
        }
    }
}

fn get_parameter_values(
    runs: &Runs, // active_runs: &Vec<String>,
    all: bool,
) -> HashMap<String, HashSet<String>> {
    let train_ids: Vec<String> = if all {
        runs.runs.keys().cloned().collect()
    } else {
        runs.active_runs.clone()
    };
    let mut param_values: HashMap<String, HashSet<String>> = train_ids
        .iter()
        .map(|train_id| runs.runs.get(train_id).unwrap())
        .map(|run| run.params.keys().cloned())
        .flatten()
        .unique()
        .map(|param_name| (param_name, HashSet::new()))
        .collect();
    for run in train_ids
        .iter()
        .map(|train_id| runs.runs.get(train_id).unwrap())
    {
        for (k, v) in &run.params {
            let values = param_values.get_mut(k).unwrap();
            values.insert(v.clone());
        }
    }
    param_values
}

async fn get_state_new(
    pool: &sqlx::postgres::PgPool,
    runs: &mut HashMap<String, Run>,
    train_ids: &Vec<String>,
    last_timestamp: &mut NaiveDateTime,
    tx_runs: Option<&Sender<HashMap<String, Run>>>,
    tx_batch_status: &Sender<(usize, usize)>,
) -> Result<(), sqlx::Error> {
    // Query to get the table structure
    // TODO: WHERE train_id not in runs.keys()
    tx_batch_status.send((0, 3));
    get_state_parameters(pool, runs).await?;
    get_state_artifacts(train_ids, pool, runs).await?;
    let mut epoch_last_timestamp = last_timestamp.clone();
    get_state_epoch_metrics(train_ids, &mut epoch_last_timestamp, pool, runs, tx_runs).await?;
    let mut every_100_last_timestamp = last_timestamp.clone();
    get_state_metrics(
        train_ids,
        &mut every_100_last_timestamp,
        pool,
        runs,
        tx_runs,
        "metrics_batch_order_100".to_string(),
    )
    .await?;
    tx_batch_status.send((1, 3));
    let mut every_10_last_timestamp = last_timestamp.clone();
    get_state_metrics(
        train_ids,
        &mut every_10_last_timestamp,
        pool,
        runs,
        tx_runs,
        "metrics_batch_order_10".to_string(),
    )
    .await?;
    tx_batch_status.send((2, 3));
    let mut batch_last_timestamp = every_10_last_timestamp.clone();
    get_state_metrics(
        train_ids,
        &mut batch_last_timestamp,
        pool,
        runs,
        tx_runs,
        "metrics_batch_order".to_string(),
    )
    .await?;
    // let mut batch_last_timestamp = every_10_last_timestamp.clone();
    get_state_metrics(
        train_ids,
        &mut batch_last_timestamp,
        pool,
        runs,
        tx_runs,
        "metrics".to_string(),
    )
    .await?;
    *last_timestamp = batch_last_timestamp.min(epoch_last_timestamp);
    tx_batch_status.send((3, 3));
    Ok(())
}

async fn get_state_metrics(
    train_ids: &Vec<String>,
    last_timestamp: &mut NaiveDateTime,
    pool: &sqlx::Pool<sqlx::Postgres>,
    runs: &mut HashMap<String, Run>,
    tx_runs: Option<&Sender<HashMap<String, Run>>>,
    metric_table: String,
) -> Result<(), sqlx::Error> {
    if train_ids.len() > 0 {
        // let last_batch_timestamp = last_timestamp.clone();
        // let mut new_batch_timestamp = last_timestamp.clone();
        let mut chunk_size = 1000i32;
        loop {
            let q = format!(
                r#"
            SELECT * FROM {}
                WHERE train_id = ANY($1) 
                    AND created_at > $2 
                ORDER BY created_at, train_id, variable, xaxis, x
                LIMIT $3
            "#,
                metric_table
            );
            let query_string = |pre: Option<String>, q: String| {
                if let Some(pre) = pre {
                    format!("{} {}", pre, q)
                } else {
                    q
                }
            };
            let q1 = query_string(None, q.clone());
            let query_time = std::time::Instant::now();
            let query = sqlx::query(q1.as_str())
                // .bind(metric_table.clone())
                .bind(train_ids)
                .bind(*last_timestamp)
                // .bind(last_id)
                .bind(chunk_size);
            // .bind(offset);
            let metric_rows = query.fetch_all(pool).await?;
            let query_elapsed_time = query_time.elapsed().as_secs_f32();
            let received_rows = metric_rows.len();
            parse_metric_rows(metric_rows, runs, last_timestamp);
            // println!(
            //     "[db] recieved batch data from {}: {}",
            //     metric_table, received_rows
            // );
            if let Some(tx) = tx_runs {
                tx.send(runs.clone()).expect("send failed");
                // println!("[db] sent {}", received_rows);
            }
            // offset += chunk_size;
            if received_rows < chunk_size as usize {
                break;
            }
            if query_elapsed_time < 0.5 {
                chunk_size *= 2;
                println!("Increased chunks: {}", chunk_size);
            } else if query_elapsed_time > 2.0 {
                chunk_size /= 2;
                println!("Decreased chunks: {}", chunk_size);
            }
        }
    }
    Ok(())
}

async fn get_state_epoch_metrics(
    train_ids: &Vec<String>,
    last_timestamp: &mut NaiveDateTime,
    pool: &sqlx::Pool<sqlx::Postgres>,
    runs: &mut HashMap<String, Run>,
    tx_runs: Option<&Sender<HashMap<String, Run>>>,
) -> Result<(), sqlx::Error> {
    // let mut new_epoch_timestamp = last_timestamp.clone();
    let chunk_size = 1000i32;
    let q = format!(
        r#"
        SELECT * FROM metrics 
            WHERE train_id = ANY($1) 
                AND created_at > $2 
                AND xaxis='epoch' 
            ORDER BY created_at, train_id, variable, xaxis, x
            LIMIT $3
            -- OFFSET $4
        "#,
    );
    loop {
        let metric_rows = sqlx::query(q.as_str())
            .bind(train_ids)
            .bind(*last_timestamp)
            .bind(chunk_size)
            // .bind(offset)
            .fetch_all(pool)
            .await?;
        let received_rows = metric_rows.len();
        parse_metric_rows(metric_rows, runs, last_timestamp);
        // *last_timestamp = batch_timestamp.max(epoch_timestamp);
        // println!("[db] recieved {}", received_rows);
        if let Some(tx) = tx_runs {
            tx.send(runs.clone()).expect("send failed");
            // println!("[db] sent {}", received_rows);
        }
        // offset += chunk_size;
        if received_rows < chunk_size as usize {
            break;
        }
    }
    Ok(())
}

async fn get_state_artifacts(
    train_ids: &Vec<String>,
    pool: &sqlx::Pool<sqlx::Postgres>,
    runs: &mut HashMap<String, Run>,
) -> Result<(), sqlx::Error> {
    let artifact_rows = sqlx::query(
        r#"
        SELECT * FROM artifacts WHERE train_id = ANY($1) ORDER BY train_id
        "#,
    )
    .bind(train_ids)
    .fetch_all(pool)
    .await?;
    Ok(
        for (train_id, rows) in &artifact_rows
            .into_iter()
            .group_by(|row| row.get::<String, _>("train_id"))
        {
            for row in rows {
                let incoming_name: String = row.try_get("name").unwrap_or_default();
                // let incoming_path: String = row.try_get("path").unwrap_or_default();
                let incoming_id: i32 = row.try_get("id").unwrap();
                if let Some(run) = runs.get_mut(&train_id) {
                    if let Some(artifact_id) = run.artifacts.get_mut(&incoming_name) {
                        // *artifact_id.path = incoming_path;
                    } else {
                        run.artifacts.insert(
                            incoming_name.clone(),
                            ArtifactId {
                                artifact_id: incoming_id,
                                train_id: train_id.clone(),
                                name: incoming_name,
                            },
                        );
                    }
                } else {
                    println!("[Artifact] No run_id {}", train_id);
                }
            }
        },
    )
}

async fn get_state_parameters(
    pool: &sqlx::Pool<sqlx::Postgres>,
    runs: &mut HashMap<String, Run>,
) -> Result<(), sqlx::Error> {
    // println!("[db] in get_state_parameters");
    let run_rows = sqlx::query(
        r#"
        SELECT * FROM runs ORDER BY train_id
        "#,
    )
    .fetch_all(pool)
    .await?;
    // println!("[db] parameters query done");
    Ok(
        for (train_id, db_params) in &run_rows
            .into_iter()
            .group_by(|row| row.get::<String, _>("train_id"))
        {
            let mut created_at: Option<chrono::NaiveDateTime> = None;

            let params: HashMap<_, _> = db_params
                .map(|row| {
                    if created_at.is_none() {
                        created_at = row.get("created_at");
                    }
                    (
                        row.get::<String, _>("variable"),
                        row.get::<String, _>("value_text"),
                    )
                })
                .collect();
            if !runs.contains_key(&train_id) {
                runs.insert(
                    train_id,
                    Run {
                        params,
                        metrics: HashMap::new(),
                        artifacts: HashMap::new(),
                        created_at: created_at.expect("No datetime for run parameters"),
                    },
                );
            }
        },
    )
}

fn parse_metric_rows(
    metric_rows: Vec<sqlx::postgres::PgRow>,
    runs: &mut HashMap<String, Run>,
    last_timestamp: &mut NaiveDateTime,
) {
    let metric_rows_sorted = metric_rows.iter().sorted_by_key(|row| {
        (
            row.get::<String, _>("train_id"),
            row.get::<String, _>("variable"),
        )
    });
    for (train_id, run_metric_rows) in &metric_rows_sorted
        // .into_iter()
        .group_by(|row| row.get::<String, _>("train_id"))
    {
        let run = runs.get_mut(&train_id).unwrap();
        for (variable, value_rows) in
            &run_metric_rows.group_by(|row| row.get::<String, _>("variable"))
        {
            let rows: Vec<_> = value_rows.collect();
            let max_timestamp = rows
                .iter()
                .max_by_key(|row| row.get::<NaiveDateTime, _>("created_at"))
                .map(|row| row.get::<NaiveDateTime, _>("created_at"));
            if let Some(max_timestamp) = max_timestamp {
                if max_timestamp > *last_timestamp {
                    *last_timestamp = max_timestamp
                }
            }
            // println!("{:?}", rows[0].columns());
            let old_orig_values = run.metrics.get(&variable);
            let orig_values: Vec<_> = rows
                .iter()
                .filter_map(|row| {
                    // println!("{:?}", row.try_get::<i32, _>("id"));
                    if let Ok(x) = row.try_get::<f64, _>("x") {
                        if let Ok(value) = row.try_get::<f64, _>("value") {
                            if let Ok(created_at) = row.try_get::<NaiveDateTime, _>("created_at") {
                                let new_value = [x, value, created_at.timestamp() as f64];
                                if let Some(old_orig_values) = old_orig_values {
                                    if old_orig_values.orig_values.contains(&new_value) {
                                        return None;
                                    }
                                }
                                return Some(new_value);
                            }
                        }
                    }
                    None
                })
                .collect();
            let xaxis = rows[0].get::<String, _>("xaxis");
            if !run.metrics.contains_key(&variable) {
                run.metrics.insert(
                    variable,
                    Metric {
                        resampled: Vec::new(),
                        orig_values,
                        xaxis,
                        values: Vec::new(),
                        last_value: *last_timestamp,
                    },
                );
            } else {
                run.metrics
                    .get_mut(&variable)
                    .unwrap()
                    .orig_values
                    .extend(orig_values);
                run.metrics
                    .get_mut(&variable)
                    .unwrap()
                    .orig_values
                    .sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
                run.metrics.get_mut(&variable).unwrap().last_value = *last_timestamp;
            }
        }
    }
}

#[derive(Debug)]
enum ArtifactTransfer {
    Done(Vec<u8>),
    Loading(usize, usize),
    Err(String),
}
// #[tokio::main(flavor = "current_thread")]
fn main() -> Result<(), sqlx::Error> {
    // Load environment variables
    // dotenv::dotenv().ok();
    let args = Args::parse();
    println!("Args: {:?}", args);
    let (tx, rx) = mpsc::channel();
    let (tx_gui_dirty, rx_gui_dirty) = mpsc::channel();
    let (tx_gui_recomputed, rx_gui_recomputed) = mpsc::channel();
    let (tx_db_filters, rx_db_filters) = mpsc::channel::<Vec<String>>();
    let rx_db_filters_am = Arc::new(std::sync::Mutex::new(rx_db_filters));
    let (tx_db_artifact, rx_db_artifact) = mpsc::channel::<ArtifactTransfer>();
    let (tx_db_artifact_path, rx_db_artifact_path) = mpsc::channel::<i32>();
    let (tx_batch_status, rx_batch_status) = mpsc::channel::<(usize, usize)>();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let rt_handle = rt.handle().clone();
    let _guard = rt.enter();

    std::thread::spawn(move || loop {
        if let Ok((gui_params, mut runs)) = rx_gui_dirty.recv() {
            recompute(&mut runs, &gui_params);
            tx_gui_recomputed
                .send(runs)
                .expect("Failed to send recomputed runs.");
        }
    });
    rt_handle.spawn(async move {
        let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
        println!("[db] connecting...");
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(&database_url)
            .await
            .expect("Can't connect to database");
        println!("[db] connected!");
        let mut train_ids: Vec<String> = Vec::new();
        let mut last_timestamp = NaiveDateTime::from_timestamp_millis(0).unwrap();
        let mut runs: HashMap<String, Run> = Default::default();
        loop {
            let loop_time = std::time::Instant::now();
            let update_params = async {
                loop {
                    {
                        let rx = rx_db_filters_am.lock().unwrap();
                        if let Ok(new_train_ids) = rx.try_recv() {
                            println!("new train ids");
                            return new_train_ids;
                        }
                    }
                    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                }
            };
            tokio::select! {
            _ = update_state(&mut runs, &pool, &train_ids, &mut last_timestamp, &tx, &tx_batch_status) => {
                tx.send(runs.clone()).expect("Failed to send data");
            },
            new_train_ids = update_params => { train_ids = new_train_ids;
                last_timestamp = NaiveDateTime::from_timestamp_millis(0).unwrap();
                runs = HashMap::new();
            }
            }
            let elapsed_loop_time = loop_time.elapsed();
            if elapsed_loop_time.as_secs_f32() < 1.0 {
                tokio::time::sleep(std::time::Duration::from_secs(1) - elapsed_loop_time).await;
            }
        }
    });
    rt_handle.spawn(async move {
        let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(&database_url)
            .await
            .expect("Can't connect to database");
        loop {
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            if let Ok(artifact_id) = rx_db_artifact_path.try_recv() {
                handle_artifact_request(artifact_id, &pool, &tx_db_artifact).await;
            }
        }
    });
    // });

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([800.0, 800.0]),
        ..Default::default()
    };
    let _ = eframe::run_native(
        "Visualizer",
        options,
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            Box::<GuiRuns>::new(GuiRuns {
                runs: Default::default(),
                dirty: true,
                db_train_runs_sender: tx_db_filters,
                db_reciever: rx,
                recomputed_reciever: rx_gui_recomputed,
                dirty_sender: tx_gui_dirty,
                initialized: false,
                data_status: DataStatus::Waiting,
                gui_params: GuiParams {
                    max_n: 1000,
                    param_filters: HashMap::new(),
                    metric_filters: HashSet::new(),
                    inspect_params: HashSet::new(),
                    n_average: 1,
                    artifact_filters: HashSet::new(),
                    time_filter: None,
                    time_filter_idx: 0,
                    x_axis: XAxis::Batch,
                    param_name_filter: "".to_string(),
                },
                // texture: None,
                artifact_handlers: HashMap::from([
                    (
                        "npy".to_string(),
                        ArtifactHandler::NPYHealPixArtifact {
                            arrays: HashMap::new(),
                            textures: HashMap::new(),
                            views: HashMap::new(),
                        },
                    ),
                    (
                        "tabular".to_string(),
                        ArtifactHandler::NPYTabularArtifact {
                            arrays: HashMap::new(),
                            views: HashMap::new(),
                        },
                    ),
                    (
                        "png".to_string(),
                        ArtifactHandler::ImageArtifact {
                            // images: HashMap::new(),
                            binaries: HashMap::new(),
                        },
                    ),
                ]),
                args,
                tx_db_artifact_path: Arc::new(Mutex::new(tx_db_artifact_path)),
                rx_db_artifact: Arc::new(Mutex::new(rx_db_artifact)),
                rx_batch_status,
                batch_status: (0, 0),
            })
        }),
    );

    Ok(())
}

async fn handle_artifact_request(
    artifact_id: i32,
    pool: &sqlx::Pool<sqlx::Postgres>,
    tx_db_artifact: &Sender<ArtifactTransfer>,
) {
    println!("[db] artifact requested: {}", artifact_id);
    let query = sqlx::query(
        r#"
                SELECT size FROM artifacts WHERE id=$1
                "#,
    )
    .bind(&artifact_id);
    // println!("{}", query);
    let size_res = query.fetch_one(pool).await;
    match size_res {
        Ok(row) => {
            let filesize = row.get::<i32, _>("size");
            println!("Got request for existing artifact id of size {}", filesize);
            // println!("[f] File {} size {}", &path, &filesize);
            let mut last_seq_num = -1;
            // let mut chunk_size = 1_000_000;
            let mut batch_size = 1;
            let mut buffer: Vec<u8> = vec![0; filesize as usize];
            let mut offset: usize = 0;
            // while offset < filesize {
            loop {
                // let length = chunk_size.min(filesize - offset);
                let query_time = std::time::Instant::now();
                let blobs_rows = sqlx::query(
                    r#"
                            SELECT seq_num, data, size FROM artifact_chunks WHERE artifact_id=$1 AND seq_num > $2 ORDER BY seq_num LIMIT $3
                            "#,
                )
                .bind(&artifact_id)
                .bind(&last_seq_num)
                .bind(&batch_size)
                .fetch_all(pool)
                .await;
                let elapsed_seconds = query_time.elapsed().as_secs_f32();
                if elapsed_seconds < 0.5 {
                    batch_size *= 2;
                } else if elapsed_seconds > 2.0 {
                    batch_size = (batch_size / 2).max(1);
                }
                // if let Ok(row) = blobs_rows {
                if let Ok(rows) = blobs_rows {
                    if rows.len() == 0 {
                        println!("[db] Fetched all available artifact chunks");
                        break;
                    }
                    println!("Got {} rows", rows.len());
                    for row in rows {
                        let chunk_size = row.get::<i32, _>("size");
                        let dst = &mut buffer[offset..offset + chunk_size as usize];
                        dst.copy_from_slice(row.get::<Vec<u8>, _>("data").as_slice());
                        offset += chunk_size as usize;
                        last_seq_num = row.get::<i32, _>("seq_num");
                        println!("[f] Read chunk {} at {}", chunk_size, offset);
                        tx_db_artifact.send(ArtifactTransfer::Loading(
                            offset as usize,
                            filesize as usize,
                        ));
                    }
                } else if let Err(error) = blobs_rows {
                    println!("error {}", error.to_string());
                    tx_db_artifact.send(ArtifactTransfer::Err(error.to_string()));
                    break;
                }
            }
            tx_db_artifact.send(ArtifactTransfer::Done(buffer));
        }
        Err(err) => {
            println!("[db] error {}", err.to_string());
            tx_db_artifact.send(ArtifactTransfer::Err(err.to_string()));
        }
    }
}

async fn get_new_runids(
    rx_db_filters: &Receiver<Vec<String>>,
    train_ids: &mut Vec<String>,
    last_timestamp: &mut NaiveDateTime,
    runs: &mut HashMap<String, Run>,
) {
    loop {
        // {
        if let Ok(new_train_ids) = rx_db_filters.try_recv() {
            //     *train_ids = new_train_ids;
            //     *last_timestamp = NaiveDateTime::from_timestamp_millis(0).unwrap();
            //     *runs = HashMap::new();
            //     return;
            //     // break;
        }
        // }
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }
}

async fn update_state(
    runs: &mut HashMap<String, Run>,
    pool: &sqlx::Pool<sqlx::Postgres>,
    train_ids: &Vec<String>,
    last_timestamp: &mut NaiveDateTime,
    tx: &Sender<HashMap<String, Run>>,
    tx_batch_status: &Sender<(usize, usize)>,
) {
    if runs.is_empty() {
        if let Err(err) = get_state_new(
            pool,
            runs,
            train_ids,
            last_timestamp,
            Some(tx),
            tx_batch_status,
        )
        .await
        {
            println!("{}", err.to_string());
        }
    } else {
        // println!("[db] parameters...");
        tx_batch_status.send((0, 3)).unwrap();
        get_state_parameters(pool, runs).await.unwrap();
        tx.send(runs.clone()).expect("Failed to send data");
        // println!("[db] artifacts...");
        get_state_artifacts(train_ids, pool, runs).await.unwrap();
        tx.send(runs.clone()).expect("Failed to send data");
        let mut epoch_last_timestamp = last_timestamp.clone();
        get_state_epoch_metrics(train_ids, &mut epoch_last_timestamp, pool, runs, Some(tx))
            .await
            .unwrap();
        // println!("[db] metrics...");
        let mut batch_every_100_last_timestamp = last_timestamp.clone();
        get_state_metrics(
            train_ids,
            &mut batch_every_100_last_timestamp,
            pool,
            runs,
            Some(tx),
            "metrics_batch_order_100".to_string(),
        )
        .await
        .unwrap();
        tx_batch_status.send((1, 3)).unwrap();
        let mut batch_every_10_last_timestamp = last_timestamp.clone();
        get_state_metrics(
            train_ids,
            &mut batch_every_10_last_timestamp,
            pool,
            runs,
            Some(tx),
            "metrics_batch_order_10".to_string(),
        )
        .await
        .unwrap();
        tx_batch_status.send((2, 3)).unwrap();
        let mut batch_last_timestamp = last_timestamp.clone();
        get_state_metrics(
            train_ids,
            &mut batch_last_timestamp,
            pool,
            runs,
            Some(tx),
            "metrics".to_string(),
        )
        .await
        .expect("Batch level metrics:");
        tx_batch_status.send((3, 3)).unwrap();
        *last_timestamp = epoch_last_timestamp.min(batch_last_timestamp);
    }
}
