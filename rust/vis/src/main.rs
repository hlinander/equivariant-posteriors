use chrono::{NaiveDateTime, Utc};
use clap::Parser;
use eframe::egui;
use eframe::egui_wgpu::{CallbackResources, ScreenDescriptor};
use egui::{epaint, Stroke};
use egui_plot::{Axis, Legend, PlotPoint, PlotPoints, Points};
use egui_plot::{Line, Plot};
use itertools::Itertools;
use ndarray::{s, IxDyn, SliceInfo, SliceInfoElem};
use sqlx::postgres::PgPoolOptions;
use sqlx::Row;
use std::borrow::Cow;
use std::rc::Rc;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use wgpu::util::DeviceExt;
// use sqlx::types::JsonValue
use glsl_layout::Uniform;
use std::collections::HashMap;
use std::collections::HashSet;
use std::env;
use std::ops::RangeInclusive;
use std::sync::mpsc::{self, Receiver, SyncSender};

pub mod np;
use colorous::{CIVIDIS, PLASMA};
use ndarray_stats::QuantileExt;
use np::load_npy_bytes;

pub mod era5;

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

#[derive(Debug, Clone)]
struct GuiParams {
    n_average: usize,
    max_n: usize,
    param_filters: HashMap<String, HashSet<String>>,
    param_name_filter: String,
    metric_filters: HashSet<String>,
    artifact_filters: HashSet<String>,
    inspect_params: HashSet<String>,
    table_sorting: HashSet<String>,
    time_filter_idx: usize,
    time_filter: Option<chrono::NaiveDateTime>,
    x_axis: XAxis,
    npy_plot_size: f64,
    render_format: wgpu::TextureFormat,
}

impl Default for GuiParams {
    fn default() -> Self {
        Self {
            npy_plot_size: 0.48,
            ..Default::default()
        }
    }
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
    artifact_type: ArtifactType,
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Clone)]
struct NPYArtifactView {
    artifact_id: ArtifactId,
    index: Vec<usize>,
    nside_div: usize,
    visualize_grid: bool,
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

#[derive(Copy, Clone)]
enum NPYArrayType {
    HealPix,
    DriscollHealy,
    Tabular,
}

struct SpatialNPYArray {
    array: NPYArray,
    array_type: NPYArrayType,
}

struct HPShader {
    render_format: wgpu::TextureFormat,
    angle1: f32,
    angle2: f32,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Uniform)]
struct HPShaderUniform {
    angle1: f32,
    angle2: f32,
}

struct HPShaderResources {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: HashMap<ArtifactId, wgpu::BindGroup>,
}

fn make_uniform_layout_entry(
    binding_num: u32,
    stage: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: binding_num,
        visibility: stage,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn upload_uniform<T: glsl_layout::Uniform>(device: &wgpu::Device, uniform: T) -> wgpu::Buffer {
    let std_140_data = uniform.std140();
    let byte_slice: &[u8] = unsafe {
        core::slice::from_raw_parts(
            (&std_140_data as *const <T as Uniform>::Std140).cast::<u8>(),
            core::mem::size_of_val::<<T as Uniform>::Std140>(&std_140_data),
        )
    };
    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        usage: wgpu::BufferUsages::UNIFORM,
        contents: byte_slice,
    });
    uniform_buf
}

impl eframe::egui_wgpu::CallbackTrait for HPShader {
    fn paint<'a>(
        &'a self,
        info: egui::PaintCallbackInfo,
        render_pass: &mut eframe::wgpu::RenderPass<'a>,
        callback_resources: &'a eframe::egui_wgpu::CallbackResources,
    ) {
        if let Some(res) = callback_resources.get::<HPShaderResources>() {
            render_pass.set_pipeline(&res.pipeline);
            render_pass.set_viewport(
                info.viewport_in_pixels().left_px as f32,
                info.viewport_in_pixels().top_px as f32,
                info.viewport_in_pixels().width_px as f32,
                info.viewport_in_pixels().height_px as f32,
                0.0,
                1.0,
            );
            // resources.bind_group.insert(
            //     bind_group,
            // );

            render_pass.set_bind_group(
                0,
                &res.bind_group
                    .get(&ArtifactId {
                        artifact_id: 0,
                        train_id: "".to_string(),
                        name: "".to_string(),
                        artifact_type: ArtifactType::NPYHealpix,
                    })
                    .unwrap(),
                &[],
            );

            render_pass.draw(0..4, 0..1);
        }
        // A command encoder executes one or many pipelines.
        // It is to WebGPU what a command buffer is to Vulkan.
    }
    fn prepare(
        &self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _screen_descriptor: &ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        // Vec::new()
        use eframe::egui_wgpu::wgpu::{PrimitiveState, VertexState};

        if callback_resources.get::<HPShaderResources>().is_none() {
            let spectrograph_vert = load_shader(device, include_bytes!("../hpshader.vertex.spv"));
            let spectrograph_frag = load_shader(device, include_bytes!("../hpshader.fragment.spv"));
            let bindings = vec![make_uniform_layout_entry(0, wgpu::ShaderStages::FRAGMENT)];
            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &bindings,
                });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                // bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
                bind_group_layouts: &[&bind_group_layout],
            });
            let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Spectrograph"),
                layout: Some(&pipeline_layout),
                vertex: VertexState {
                    module: &spectrograph_vert,
                    entry_point: "main",
                    buffers: &[],
                },
                primitive: PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &spectrograph_frag,
                    entry_point: "main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.render_format,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::COLOR,
                    })],
                }),
                multiview: None,
            });
            callback_resources.insert(HPShaderResources {
                pipeline: render_pipeline,
                bind_group: [].into(),
                bind_group_layout,
                // bind_group: [((self.node, self.out_port), bind_group)].into(),
            });
            let uniform_buf = upload_uniform(
                device,
                HPShaderUniform {
                    angle1: self.angle1,
                    angle2: self.angle2,
                },
            );
            let resources = callback_resources.get::<HPShaderResources>().unwrap();
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &resources.bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(uniform_buf.as_entire_buffer_binding()),
                }],
            });
            let resources = callback_resources.get_mut::<HPShaderResources>().unwrap();
            resources.bind_group.insert(
                ArtifactId {
                    artifact_id: 0,
                    train_id: "".to_string(),
                    name: "".to_string(),
                    artifact_type: ArtifactType::NPYHealpix,
                },
                bind_group,
            );
        } else {
            let uniform_buf = upload_uniform(
                device,
                HPShaderUniform {
                    angle1: self.angle1,
                    angle2: self.angle2,
                },
            );
            let resources = callback_resources.get::<HPShaderResources>().unwrap();
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &resources.bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(uniform_buf.as_entire_buffer_binding()),
                }],
            });
            let resources = callback_resources.get_mut::<HPShaderResources>().unwrap();
            if let Some(bg) = resources.bind_group.get_mut(&ArtifactId {
                artifact_id: 0,
                train_id: "".to_string(),
                name: "".to_string(),
                artifact_type: ArtifactType::NPYHealpix,
            }) {
                *bg = bind_group;
            }
        }

        Vec::new()
    }

    fn finish_prepare(
        &self,
        _device: &eframe::wgpu::Device,
        _queue: &eframe::wgpu::Queue,
        _egui_encoder: &mut eframe::wgpu::CommandEncoder,
        _callback_resources: &mut eframe::egui_wgpu::CallbackResources,
    ) -> Vec<eframe::wgpu::CommandBuffer> {
        Vec::new()
    }
}

fn load_shader(device: &wgpu::Device, bytes: &'static [u8]) -> wgpu::ShaderModule {
    let ints = unsafe {
        core::slice::from_raw_parts(
            bytes.as_ptr().cast(),
            bytes.len() / core::mem::size_of::<u32>(),
        )
    };
    if device
        .features()
        .contains(wgpu::Features::SPIRV_SHADER_PASSTHROUGH)
    {
        let source = wgpu::util::make_spirv_raw(bytes);
        unsafe {
            device.create_shader_module_spirv(&wgpu::ShaderModuleDescriptorSpirV {
                label: None,
                source,
            })
        }
    } else {
        // unsafe {
        //     device.create_shader_module_unchecked(wgpu::ShaderModuleDescriptor {
        //         label: None,
        //         source: wgpu::ShaderSource::SpirV(ints.into()),
        //     })
        // }
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::SpirV(ints.into()),
        })
    }
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
    tx_path_mutex: Arc<Mutex<SyncSender<i32>>>,
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
            } else {
                // if let Ok(download_status) = download_progress.rx_update.try_recv() {
                //     download_progress.status = download_status;
                // }
                while let Ok(download_status) = download_progress.rx_update.try_recv() {
                    download_progress.status = download_status;
                }
            }
        }
        BinaryArtifact::Loaded(_) => {}
        BinaryArtifact::Error(_) => {}
    }
    if let Some(new_binary_artifact) = new_binary_artifact {
        *binary_artifact = new_binary_artifact;
    }
}

#[derive(PartialEq, Eq, Hash)]
enum ArtifactHandlerType {
    SpatialNPY,
    TabularNPY,
    Image,
    Unknown,
}

enum ArtifactHandler {
    NPYArtifact {
        textures: HashMap<NPYArtifactView, ColorTextureInfo>,
        arrays: HashMap<ArtifactId, SpatialNPYArray>,
        views: HashMap<ArtifactId, NPYArtifactView>,
        colormap_artifacts: HashSet<ArtifactId>, // container: NPYVisContainer,
        hover_lon: f64,
        hover_lat: f64,
    },
    // NPYDriscollHealyArtifact {
    //     container: NPYVisContainer,
    // },
    NPYTabularArtifact {
        arrays: HashMap<ArtifactId, SpatialNPYArray>,
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
    tx_path_mutex: &mut Arc<Mutex<SyncSender<i32>>>,
    rx_artifact_mutex: &Arc<Mutex<Receiver<ArtifactTransfer>>>,
) {
    match handler {
        ArtifactHandler::NPYArtifact {
            textures,
            arrays,
            views,
            colormap_artifacts,
            hover_lon,
            hover_lat,
        } => handle_add_npy(arrays, &artifact_id, tx_path_mutex, rx_artifact_mutex),
        // ArtifactHandler::NPYDriscollHealyArtifact { container } => handle_add_npy(
        //     &mut container.arrays,
        //     &artifact_id,
        //     tx_path_mutex,
        //     rx_artifact_mutex,
        // ),
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
    arrays: &mut HashMap<ArtifactId, SpatialNPYArray>,
    artifact_id: &ArtifactId,
    tx_path_mutex: &mut Arc<Mutex<SyncSender<i32>>>,
    rx_artifact_mutex: &Arc<Mutex<Receiver<ArtifactTransfer>>>,
) {
    match arrays.get_mut(artifact_id) {
        None => {
            arrays.insert(
                artifact_id.clone(),
                SpatialNPYArray {
                    array: NPYArray::Loading(download_artifact(
                        artifact_id.clone(),
                        tx_path_mutex.clone(),
                        rx_artifact_mutex.clone(),
                    )),
                    array_type: match artifact_id.artifact_type {
                        ArtifactType::NPYHealpix => NPYArrayType::HealPix,
                        ArtifactType::NPYDriscollHealy => NPYArrayType::DriscollHealy,
                        _ => todo!(),
                    },
                },
            );
        }
        Some(npyarray) => {
            let mut new_npyarray = None;
            let SpatialNPYArray { array, array_type } = npyarray;
            match array {
                NPYArray::Loading(binary_artifact) => {
                    poll_artifact_download(binary_artifact);
                    match binary_artifact {
                        BinaryArtifact::Loading(_) => {}
                        BinaryArtifact::Loaded(binary_data) => match load_npy_bytes(&binary_data) {
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
                *npyarray = SpatialNPYArray {
                    array: new_npyarray,
                    array_type: *array_type,
                };
            }
        }
    }
}

struct ColorImageInfo {
    image: egui::ColorImage,
    min_val: f64,
    max_val: f64,
}

fn image_from_ndarray_healpix(
    array: &ndarray::prelude::ArrayBase<
        ndarray::OwnedRepr<f32>,
        ndarray::prelude::Dim<ndarray::IxDynImpl>,
    >,
    view: &NPYArtifactView,
) -> ColorImageInfo {
    let width = 1000;
    let height = 1000;
    let mut img = egui::ColorImage::new([width, height], egui::Color32::WHITE);
    let mut local_index = vec![0; array.shape().len()];
    for (dim_idx, dim) in view.index.iter().enumerate() {
        local_index[dim_idx] = *dim;
    }
    // let mut local_index = view.index.iter().map(|dim| dim)
    let (max, min) = min_max_hp(view, array);
    let t = |x: f32| (x - min) / (max - min);
    let ndim = local_index.len();
    let nside = ((array.shape().last().unwrap() / 12) as f32).sqrt() as u32;
    let depth = cdshealpix::depth(nside);
    for y in 0..height {
        for x in 0..width {
            let (lon, lat) = (
                x as f64 / width as f64 * 2.0 * std::f64::consts::PI,
                -(y as f64 - height as f64 / 2.0) / (height as f64 / 2.0) * std::f64::consts::PI
                    / 2.0,
            );
            let hp_idx = cdshealpix::nested::hash(depth, lon, lat);
            local_index[ndim - 1] = hp_idx as usize;
            let v = t(*array.get(local_index.as_slice()).unwrap()) as f64;
            let color: colorous::Color = match v {
                x if x < 0.0 => colorous::Color { r: 0, g: 255, b: 0 },
                x if x > 1.0 => colorous::Color { r: 0, g: 0, b: 255 },
                _ => PLASMA.eval_continuous(v),
            };
            img.pixels[y * width + x] = egui::Color32::from_rgb(color.r, color.g, color.b);
        }
    }
    // Visualize the borders of the windows (window_size = 16 => nside_window = nside / sqrt(16))
    if view.visualize_grid {
        let vis_depth =
            cdshealpix::depth(nside / (2_u32.pow(view.nside_div as u32)).clamp(1, nside));
        let n_vis_pixels = cdshealpix::n_hash(vis_depth);
        for pixel_hash in 0..n_vis_pixels {
            let lonlats = cdshealpix::nested::path_along_cell_edge(
                vis_depth,
                pixel_hash,
                &cdshealpix::compass_point::Cardinal::N,
                true,
                100,
            );
            for (lon, lat) in &*lonlats {
                let x = lon / (2.0 * std::f64::consts::PI) * width as f64;
                let y = (lat + std::f64::consts::PI / 2.0) / std::f64::consts::PI * height as f64;
                let xu = (x as usize).clamp(0, width - 1);
                let yu = (y as usize).clamp(0, height - 1);
                // println!("{}, {}", xu, yu);
                // for xu in xu - 1..=xu + 1 {
                // for yu in yu - 1..=yu + 1 {
                let xu = xu.clamp(0, width - 1);
                let yu = yu.clamp(0, height - 1);
                img.pixels[yu * width + xu] = egui::Color32::from_rgb(0, 255, 0);
                // }
                // }
            }
        }
    }
    ColorImageInfo {
        image: img,
        min_val: min as f64,
        max_val: max as f64,
    }
}

fn sample_hp_array(
    array: &ndarray::prelude::ArrayBase<
        ndarray::OwnedRepr<f32>,
        ndarray::prelude::Dim<ndarray::IxDynImpl>,
    >,
    view: &NPYArtifactView,
    lon: f64,
    lat: f64,
) -> f32 {
    let mut local_index = view
        .index
        .iter()
        .cloned()
        .map(|dim| dim)
        .chain(std::iter::once(0))
        .collect_vec();

    let nside = ((array.shape().last().unwrap() / 12) as f32).sqrt() as u32;
    let depth = cdshealpix::depth(nside);
    // let (lon, lat) = (
    //     x as f64 / width as f64 * 2.0 * std::f64::consts::PI,
    //     -(y as f64 - height as f64 / 2.0) / (height as f64 / 2.0) * std::f64::consts::PI / 2.0,
    // );
    let hp_idx = cdshealpix::nested::hash(depth, lon, lat);
    let ndim = local_index.len();
    local_index[ndim - 1] = hp_idx as usize;
    // println!("{:?}", local_index);
    let v = *array.get(local_index.as_slice()).unwrap() as f32;
    v
}

fn min_max_hp(
    view: &NPYArtifactView,
    array: &ndarray::prelude::ArrayBase<
        ndarray::OwnedRepr<f32>,
        ndarray::prelude::Dim<ndarray::IxDynImpl>,
    >,
) -> (f32, f32) {
    let si = unsafe {
        SliceInfo::<_, IxDyn, IxDyn>::new(
            view.index
                .iter()
                .map(|i| SliceInfoElem::Index(*i as isize))
                .chain([SliceInfoElem::Slice {
                    start: 0,
                    end: None,
                    step: 1,
                }])
                .collect_vec(),
        )
        .unwrap()
    };
    let aslice = array.slice(&si);
    let max = aslice.max().unwrap();
    let min = aslice.min().unwrap();
    (*max, *min)
}

fn image_from_ndarray_driscoll_healy(
    array: &ndarray::prelude::ArrayBase<
        ndarray::OwnedRepr<f32>,
        ndarray::prelude::Dim<ndarray::IxDynImpl>,
    >,
    view: &NPYArtifactView,
    min: f32,
    max: f32,
) -> ColorImageInfo {
    let width = 1000;
    let height = 1000;
    let mut img = egui::ColorImage::new([width, height], egui::Color32::WHITE);
    let mut local_index = vec![0; array.shape().len()];
    for (dim_idx, dim) in view.index.iter().enumerate() {
        local_index[dim_idx] = *dim;
    }
    // let (max, min) = min_max_driscoll_healy(view, array);
    let t = |x: f32| (x - min) / (max - min);
    let ndim = local_index.len();
    for y in 0..height {
        for x in 0..width {
            let (lon, lat) = (
                x as f64 / width as f64 * 2.0 * std::f64::consts::PI,
                (y as f64 - height as f64 / 2.0) / (height as f64 / 2.0) * std::f64::consts::PI
                    / 2.0,
            );

            let lon_idx_f = lon * array.shape()[ndim - 1] as f64 / (2.0 * std::f64::consts::PI);
            let lon_idx = (lon_idx_f as usize).clamp(0, array.shape()[ndim - 1] - 1);

            let lat_idx_f = (lat + std::f64::consts::PI / 2.0) * array.shape()[ndim - 2] as f64
                / std::f64::consts::PI;
            let lat_idx = (lat_idx_f as usize).clamp(0, array.shape()[ndim - 2] - 1);
            // let nside = ((array.shape().last().unwrap() / 12) as f32).sqrt() as u32;
            // let depth = cdshealpix::depth(nside);
            // let hp_idx = cdshealpix::nested::hash(depth, lon, lat);
            // if hp_idx < cdshealpix::n_hash(depth) {
            // println!("nside {}, depth {}, idx {}", nside, depth, hp_idx);
            // dbg!(array.shape());
            local_index[ndim - 1] = lon_idx;
            local_index[ndim - 2] = lat_idx;
            // dbg!(&local_index, array.shape());
            // let color =
            //     PLASMA.eval_continuous(t(*array.get(local_index.as_slice()).unwrap()) as f64);
            let v = t(*array.get(local_index.as_slice()).unwrap()) as f64;
            let color: colorous::Color = match v {
                x if x < 0.0 => colorous::Color { r: 0, g: 255, b: 0 },
                x if x > 1.0 => colorous::Color { r: 0, g: 0, b: 255 },
                _ => PLASMA.eval_continuous(v),
            };
            img.pixels[y * width + x] = egui::Color32::from_rgb(color.r, color.g, color.b);
            // }
        }
    }
    ColorImageInfo {
        image: img,
        min_val: min as f64,
        max_val: max as f64,
    }
}

fn min_max_driscoll_healy(
    view: &NPYArtifactView,
    array: &ndarray::prelude::ArrayBase<
        ndarray::OwnedRepr<f32>,
        ndarray::prelude::Dim<ndarray::IxDynImpl>,
    >,
) -> (f32, f32) {
    let si = unsafe {
        SliceInfo::<_, IxDyn, IxDyn>::new(
            view.index
                .iter()
                .map(|i| SliceInfoElem::Index(*i as isize))
                .chain([
                    SliceInfoElem::Slice {
                        start: 0,
                        end: None,
                        step: 1,
                    },
                    SliceInfoElem::Slice {
                        start: 0,
                        end: None,
                        step: 1,
                    },
                ])
                .collect_vec(),
        )
        .unwrap()
    };
    let aslice = array.slice(&si);
    let max = aslice.max().unwrap();
    let min = aslice.min().unwrap();
    (*min, *max)
}
struct GuiRuns {
    runs: Runs,
    // dirty: bool,
    db_train_runs_sender: SyncSender<Vec<String>>,
    db_train_runs_sender_slot: Option<Vec<String>>,
    gui_params_sender: SyncSender<(GuiParams, HashMap<String, Run>)>,
    gui_params_sender_slot: Option<(GuiParams, HashMap<String, Run>)>,
    db_reciever: Receiver<HashMap<String, Run>>,
    recomputed_reciever: Receiver<HashMap<String, Run>>,
    tx_db_artifact_path: Arc<Mutex<SyncSender<i32>>>,
    rx_db_artifact: Arc<Mutex<Receiver<ArtifactTransfer>>>,
    rx_batch_status: Receiver<(usize, usize)>,
    batch_status: (usize, usize),
    initialized: bool,
    data_status: DataStatus,
    gui_params: GuiParams,
    table_active: bool,
    artifact_handlers: HashMap<ArtifactHandlerType, ArtifactHandler>,
    artifact_dispatch: HashMap<ArtifactType, ArtifactHandlerType>,
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
            ctx.set_zoom_factor(1.0);
        }
        if let Some(gui_params_package) = self.gui_params_sender_slot.take() {
            if self
                .gui_params_sender
                .try_send(gui_params_package.clone())
                .is_err()
            {
                self.gui_params_sender_slot = Some(gui_params_package);
            }
        }
        if let Some(db_train_runs_package) = self.db_train_runs_sender_slot.take() {
            if self
                .db_train_runs_sender
                .try_send(db_train_runs_package.clone())
                .is_err()
            {
                self.db_train_runs_sender_slot = Some(db_train_runs_package);
            }
        }
        while let Ok(new_runs) = self.recomputed_reciever.try_recv() {
            self.runs.runs = new_runs;
            if self.runs.runs.len() > 0 && self.data_status == DataStatus::FirstDataArrived {
                self.data_status = DataStatus::FirstDataProcessed;
            }
        }
        while let Ok(new_runs) = self.db_reciever.try_recv() {
            for train_id in new_runs.keys() {
                if !self.runs.runs.contains_key(train_id) {
                    let new_active = get_train_ids_from_filter(&new_runs, &self.gui_params);
                    println!("[app] recieved new runs, sending to compute...");
                    self.db_train_runs_sender_slot = Some(new_active);
                    // self.db_train_runs_sender.try_send(new_active);
                    // .expect("Failed to send train runs to db thread");
                    break;
                }
            }
            self.gui_params_sender_slot = Some((self.gui_params.clone(), new_runs));
            if self.data_status == DataStatus::Waiting {
                self.data_status = DataStatus::FirstDataArrived;
            }
            // self.recompute();
        }
        // return;

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
            let (resp, painter) = ui.allocate_painter(
                egui::Vec2::new(300.0, 300.0),
                egui::Sense::focusable_noninteractive(),
            );
            let angle1 = if let Some(pos) = ctx.pointer_latest_pos() {
                pos.x / 500.0
            } else {
                0.0
            };
            let angle2 = if let Some(pos) = ctx.pointer_latest_pos() {
                pos.y / 500.0
            } else {
                0.0
            };
            painter.add(egui::Shape::Callback(
                eframe::egui_wgpu::Callback::new_paint_callback(
                    resp.rect,
                    HPShader {
                        render_format: self.gui_params.render_format,
                        angle1,
                        angle2, // node: *node_idx,
                                // out_port: output_id,
                    },
                ),
            ));

            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    if ui.button("Time").clicked() {
                        self.gui_params.x_axis = XAxis::Time;
                    }
                    if ui.button("Batch").clicked() {
                        self.gui_params.x_axis = XAxis::Batch;
                    }
                    self.render_time_selector(ui);
                    while let Ok(batch_status) = self.rx_batch_status.try_recv() {
                        self.batch_status = batch_status;
                    }
                });
                if self.batch_status.1 > 0 {
                    ui.add(
                        egui::ProgressBar::new(
                            self.batch_status.0 as f32 / self.batch_status.1 as f32,
                        )
                        .desired_width(50.0),
                    );
                }
            });

            ui.separator();
            egui::ScrollArea::vertical()
                .id_source("central_space")
                .show(ui, |ui| {
                    let collapsing = ui.collapsing("Tabular", |ui| {
                        self.render_table(ui, &run_ensemble_color);
                    });
                    self.table_active = collapsing.fully_open();
                    self.render_artifacts(ui, &run_ensemble_color);
                    self.render_plots(ui, metric_names, &run_ensemble_color);
                });
        });
        self.initialized = true;
        ctx.request_repaint();
        // ctx.repaint_causes()
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Ord, PartialOrd, Debug)]
enum ArtifactType {
    PngImage,
    NPYHealpix,
    NPYDriscollHealy,
    NPYTabular,
    Unknown,
}

fn get_artifact_type(path: &String) -> ArtifactType {
    match path.split(".").last().unwrap_or("unknown") {
        "npy" => ArtifactType::NPYHealpix,
        "npydh" => ArtifactType::NPYDriscollHealy,
        "tabular" => ArtifactType::NPYTabular,
        "png" => ArtifactType::PngImage,
        _ => ArtifactType::Unknown,
    }
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
    gui_params: &mut GuiParams,
    runs: &HashMap<String, Run>,
    filtered_runs: &Vec<String>,
    run_ensemble_color: &HashMap<String, egui::Color32>,
) {
    match handler {
        ArtifactHandler::NPYArtifact {
            textures,
            arrays,
            views,
            colormap_artifacts,
            hover_lon,
            hover_lat,
        } => {
            // let texture = texture.get_or_insert_with(|| {});
            let mut to_remove = Vec::new();
            let npy_axis_id = ui.id().with("npy_axis");
            let available_artifact_names: Vec<&String> = arrays.keys().map(|id| &id.name).collect();
            // let mut to_be_reloaded = None;
            {
                for (artifact_name, filtered_arrays) in gui_params
                    .artifact_filters
                    .iter()
                    .filter(|name| available_artifact_names.contains(name))
                    .map(|name| {
                        (
                            name,
                            arrays.iter().filter(|(key, _v)| {
                                key.name == *name && filtered_runs.contains(&key.train_id)
                            }), // .collect_vec(),
                        )
                    })
                {
                    // artifact_name,
                    ui.add(egui::Slider::new(&mut gui_params.npy_plot_size, 0.0..=1.0));
                    let plot_width = ui.available_width() * gui_params.npy_plot_size as f32;
                    let plot_height = ui.available_width() * gui_params.npy_plot_size as f32 * 0.5;
                    ui.horizontal_wrapped(|ui| {
                        for (artifact_name, array_group) in filtered_arrays
                            .group_by(|(aid, _)| aid.name.clone())
                            .into_iter()
                            .sorted_by_key(|(artifact_name, _)| artifact_name.clone())
                        // .into_iter()
                        // .sorted_by_key(|(name, _)| name)
                        {
                            ui.group(|ui| {
                                ui.label(egui::RichText::new(artifact_name).size(20.0));
                                ui.allocate_space(egui::Vec2::new(ui.available_width(), 0.0));
                                for (artifact_id, array) in array_group {
                                    if ui.button("reload").clicked() {
                                        to_remove.push(artifact_id.clone());
                                    }
                                    match &array.array {
                                        NPYArray::Loading(binary_artifact) => {
                                            render_artifact_download_progress(&binary_artifact, ui);
                                        }
                                        NPYArray::Loaded(npyarray) => {
                                            // ui.allocate_ui()
                                            // if ui.button("reload").clicked() {
                                            // to_be_reloaded = Some(artifact_id);
                                            // array.array
                                            // }
                                            ui.allocate_ui(
                                                egui::Vec2::from([plot_width, plot_height + 200.0]),
                                                |ui| match array.array_type {
                                                    NPYArrayType::HealPix => {
                                                        render_npy_artifact_hp(
                                                            ui,
                                                            runs,
                                                            artifact_id,
                                                            gui_params,
                                                            run_ensemble_color,
                                                            views,
                                                            &npyarray,
                                                            textures,
                                                            plot_width,
                                                            npy_axis_id,
                                                            hover_lon,
                                                            hover_lat,
                                                        )
                                                    }
                                                    NPYArrayType::DriscollHealy => {
                                                        render_npy_artifact_driscoll_healy(
                                                            ui,
                                                            runs,
                                                            artifact_id,
                                                            gui_params,
                                                            run_ensemble_color,
                                                            views,
                                                            &npyarray,
                                                            textures,
                                                            plot_width,
                                                            npy_axis_id,
                                                        )
                                                    }
                                                    NPYArrayType::Tabular => todo!(),
                                                },
                                            );
                                        }
                                        NPYArray::Error(err) => {
                                            ui.label(err);
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
            for artifact_id in to_remove {
                arrays.remove(&artifact_id);
                textures.retain(|k, _| k.artifact_id != artifact_id);
            }
            // if let Some(artifact_id) = to_be_reloaded {
            // arrays.remove(artifact_id);
            // }
        }
        // ArtifactHandler::NPYDriscollHealyArtifact { container } => {
        //     // let texture = texture.get_or_insert_with(|| {});
        //     let npy_axis_id = ui.id().with("npy_axis");
        //     let available_artifact_names: Vec<&String> =
        //         container.arrays.keys().map(|id| &id.name).collect();
        //     for (artifact_name, filtered_arrays) in gui_params
        //         .artifact_filters
        //         .iter()
        //         .filter(|name| available_artifact_names.contains(name))
        //         .map(|name| {
        //             (
        //                 name,
        //                 container.arrays.iter().filter(|(key, _v)| {
        //                     key.name == *name && filtered_runs.contains(&key.train_id)
        //                 }),
        //             )
        //         })
        //     {
        //         // artifact_name,
        //         ui.add(egui::Slider::new(&mut gui_params.npy_plot_size, 0.0..=1.0));
        //         let plot_width = ui.available_width() * gui_params.npy_plot_size as f32;
        //         let plot_height = ui.available_width() * gui_params.npy_plot_size as f32 * 0.5;
        //         ui.horizontal_wrapped(|ui| {
        //             for (artifact_name, array_group) in
        //                 &filtered_arrays.group_by(|(aid, _)| aid.name.clone())
        //             {
        //                 ui.group(|ui| {
        //                     ui.label(egui::RichText::new(artifact_name).size(20.0));
        //                     ui.allocate_space(egui::Vec2::new(ui.available_width(), 0.0));
        //                     for (artifact_id, array) in array_group {
        //                         match array {
        //                             NPYArray::Loading(binary_artifact) => {
        //                                 render_artifact_download_progress(binary_artifact, ui);
        //                             }
        //                             NPYArray::Loaded(array) => {
        //                                 // ui.allocate_ui()
        //                                 ui.allocate_ui(
        //                                     egui::Vec2::from([plot_width, plot_height + 200.0]),
        //                                     |ui| {
        //                                         render_npy_artifact_driscoll_healy(
        //                                             ui,
        //                                             runs,
        //                                             artifact_id,
        //                                             gui_params,
        //                                             run_ensemble_color,
        //                                             &mut container.views,
        //                                             array,
        //                                             &mut container.textures,
        //                                             plot_width,
        //                                             npy_axis_id,
        //                                         )
        //                                     },
        //                                 );
        //                             }
        //                             NPYArray::Error(err) => {
        //                                 ui.label(err);
        //                                 // ui.colored_label(egui::Color32::RED, err);
        //                                 // ui.allocate_space(egui::Vec2::new(ui.available_width(), 0.0));
        //                             }
        //                         }
        //                     }
        //                 });
        //             }
        //         });
        //     }
        // }
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
                    match &array.array {
                        NPYArray::Loading(binary_artifact) => {
                            render_artifact_download_progress(&binary_artifact, ui);
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
                                        &array,
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
        } // ArtifactHandler::NPYDriscollHealyArtifact { container } => todo!(),
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

struct ColorTextureInfo {
    texture_handle: egui::TextureHandle,
    min_val: f64,
    max_val: f64,
}

fn render_npy_artifact_hp(
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
    // textures: &mut HashMap<NPYArtifactView, egui::TextureHandle>,
    textures: &mut HashMap<NPYArtifactView, ColorTextureInfo>,
    plot_width: f32,
    npy_axis_id: egui::Id,
    hover_lon: &mut f64,
    hover_lat: &mut f64,
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
            nside_div: 4,
            visualize_grid: false,
        });
        // let surface_names = vec![vec!["batch"], vec!["msl", "u10", "v10", "t2m"]];
        // ui.checkbox(, )
        let era5_meta = era5::era5_meta();
        for (dim_idx, dim) in array
            .shape()
            .iter()
            .enumerate()
            .take(array.shape().len() - 1)
        {
            ui.spacing_mut().slider_width = 300.0; // ui.available_width() - 300.0;
            ui.checkbox(&mut view.visualize_grid, "Show grid");
            if view.visualize_grid {
                ui.add(egui::Slider::new(&mut view.nside_div, 1..=16));
            }
            ui.add(
                egui::Slider::new(&mut view.index[dim_idx], 0..=(dim - 1)).custom_formatter(
                    |index_f, _| match array.shape().len() {
                        2 => {
                            if dim_idx == 0 && array.shape()[1] > 128 {
                                let idx =
                                    (index_f as usize).clamp(0, era5_meta.surface.names.len() - 1);
                                return format!(
                                    "{} [{}], {} [{}]",
                                    era5_meta.surface.names[idx],
                                    era5_meta.surface.units[idx],
                                    era5_meta.surface.long_names[idx],
                                    index_f as usize,
                                );
                            } else {
                                return format!("{}", index_f as usize);
                            }
                        }
                        3 => {
                            if array.shape()[0] == 5 {
                                match dim_idx {
                                    0 => {
                                        return format!(
                                            "{} [{}], {}",
                                            era5_meta.upper.names[index_f as usize],
                                            era5_meta.upper.units[index_f as usize],
                                            era5_meta.upper.long_names[index_f as usize],
                                        );
                                    }
                                    1 => {
                                        return format!(
                                            "{} [{}], {}",
                                            era5_meta.upper.levels[index_f as usize],
                                            era5_meta.upper.level_units,
                                            era5_meta.upper.level_name,
                                        );
                                    }
                                    _ => {
                                        return format!("{}", index_f as usize);
                                    }
                                }
                            }
                            if dim_idx == 1 {
                                return format!(
                                    "{} [{}], {}",
                                    era5_meta.surface.names[index_f as usize],
                                    era5_meta.surface.units[index_f as usize],
                                    era5_meta.surface.long_names[index_f as usize],
                                );
                            } else {
                                return format!("{}", index_f as usize);
                            }
                        }
                        4 => match dim_idx {
                            1 => {
                                return format!(
                                    "{} [{}], {}",
                                    era5_meta.upper.names[index_f as usize],
                                    era5_meta.upper.units[index_f as usize],
                                    era5_meta.upper.long_names[index_f as usize],
                                );
                            }
                            2 => {
                                return format!(
                                    "{} [{}], {}",
                                    era5_meta.upper.levels[index_f as usize],
                                    era5_meta.upper.level_units,
                                    era5_meta.upper.level_name,
                                );
                            }
                            _ => {
                                return format!("{}", index_f as usize);
                            }
                        },
                        _ => return "na".to_string(),
                    },
                ),
            );
        }
        ui.label(array.shape().iter().map(|x| x.to_string()).join(","));
        if !textures.contains_key(&view) {
            let mut texture = ui.ctx().load_texture(
                &artifact_id.name,
                egui::ColorImage::example(),
                egui::TextureOptions::default(),
            );
            let color_img_info = image_from_ndarray_healpix(array, view);
            texture.set(color_img_info.image, egui::TextureOptions::default());
            textures.insert(
                view.clone(),
                ColorTextureInfo {
                    texture_handle: texture,
                    min_val: color_img_info.min_val,
                    max_val: color_img_info.max_val,
                },
            );
        }
        let pi = egui_plot::PlotImage::new(
            textures.get(&view).unwrap().texture_handle.id(),
            // texture.id(),
            egui_plot::PlotPoint::from([0.0, 0.0]),
            [2.0 * 3.14, 3.14],
        );
        // texture.set(img, egui::TextureOptions::default());
        // ui.image((texture.id(), texture.size_vec2()));
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
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
                        if let Some(pos) = plot_ui.pointer_coordinate() {
                            if pos.x > -std::f64::consts::PI && pos.x < std::f64::consts::PI {
                                if pos.y > -std::f64::consts::PI / 2.0
                                    && pos.y < std::f64::consts::PI / 2.0
                                {
                                    *hover_lon = pos.x + std::f64::consts::PI;
                                    *hover_lat = pos.y;
                                    // let ps = PlotPoints::new(vec![[pos.x, pos.y]]);
                                    // plot_ui.points(
                                    //     Points::new(ps).radius(5.0).color(egui::Color32::GREEN),
                                    // );
                                    // let mut value_hover_pos = pos.clone();
                                    // value_hover_pos.y += plot_ui.transform()
                                    // plot_ui.text(egui_plot::Text::new(
                                    // pos,
                                    // egui::RichText::new(format!("{:02}", v)).size(10.0),
                                    // ));
                                }
                            }
                        }
                    });

                let color_info = textures.get(&view).unwrap();
                let t_and_color = (0..50)
                    .map(|x| x as f64 / 50.0)
                    .map(|t| (t, PLASMA.eval_continuous(t as f64)));
                let lines = t_and_color.tuple_windows().map(|((t1, c1), (t2, c2))| {
                    let color = egui::Color32::from_rgb(c1.r, c1.g, c1.b);
                    let v1 = color_info.min_val + t1 * (color_info.max_val - color_info.min_val);
                    let v2 = color_info.min_val + t2 * (color_info.max_val - color_info.min_val);
                    egui_plot::Line::new([[0.0, v1], [0.0, v2]].into_iter().collect_vec())
                        .color(color)
                        .width(10.0)
                });
                Plot::new((artifact_id, "colorbar"))
                    .width(90.0)
                    .height(plot_width / 2.0)
                    .auto_bounds(egui::Vec2b::TRUE)
                    .y_axis_position(egui_plot::HPlacement::Right)
                    // .grid_spacing()
                    .show(ui, |plot_ui| {
                        for line in lines {
                            plot_ui.line(line)
                        }
                    });
            });
            let v = sample_hp_array(array, view, *hover_lon, *hover_lat);
            ui.label(format!("{}", v));
        });
    });
}
fn render_npy_artifact_driscoll_healy(
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
    // textures: &mut HashMap<NPYArtifactView, egui::TextureHandle>,
    textures: &mut HashMap<NPYArtifactView, ColorTextureInfo>,
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
            index: vec![0; array.shape().len() - 2],
            nside_div: 4,
            visualize_grid: false,
        });
        // let surface_names = vec![vec!["batch"], vec!["msl", "u10", "v10", "t2m"]];
        let era5_meta = era5::era5_meta();
        for (dim_idx, dim) in array
            .shape()
            .iter()
            .enumerate()
            .take(array.shape().len() - 2)
        {
            ui.add(
                egui::Slider::new(&mut view.index[dim_idx], 0..=(dim - 1)).custom_formatter(
                    |index_f, _| match array.shape().len() {
                        3 => {
                            if dim_idx == 0 && array.shape()[1] > 128 {
                                return format!(
                                    "{} [{}], {}",
                                    era5_meta.surface.names[index_f as usize],
                                    era5_meta.surface.units[index_f as usize],
                                    era5_meta.surface.long_names[index_f as usize],
                                );
                            } else {
                                return format!("{}", index_f as usize);
                            }
                        }
                        4 => {
                            if array.shape()[0] == 5 {
                                match dim_idx {
                                    0 => {
                                        return format!(
                                            "{} [{}], {}",
                                            era5_meta.upper.names[index_f as usize],
                                            era5_meta.upper.units[index_f as usize],
                                            era5_meta.upper.long_names[index_f as usize],
                                        );
                                    }
                                    1 => {
                                        return format!(
                                            "{} [{}], {}",
                                            era5_meta.upper.levels[index_f as usize],
                                            era5_meta.upper.level_units,
                                            era5_meta.upper.level_name,
                                        );
                                    }
                                    _ => {
                                        return format!("{}", index_f as usize);
                                    }
                                }
                            }
                            if dim_idx == 1 {
                                return format!(
                                    "{} [{}], {}",
                                    era5_meta.surface.names[index_f as usize],
                                    era5_meta.surface.units[index_f as usize],
                                    era5_meta.surface.long_names[index_f as usize],
                                );
                            } else {
                                return format!("{}", index_f as usize);
                            }
                        }
                        5 => match dim_idx {
                            1 => {
                                return format!(
                                    "{} [{}], {}",
                                    era5_meta.upper.names[index_f as usize],
                                    era5_meta.upper.units[index_f as usize],
                                    era5_meta.upper.long_names[index_f as usize],
                                );
                            }
                            2 => {
                                return format!(
                                    "{} [{}], {}",
                                    era5_meta.upper.levels[index_f as usize],
                                    era5_meta.upper.level_units,
                                    era5_meta.upper.level_name,
                                );
                            }
                            _ => {
                                return format!("{}", index_f as usize);
                            }
                        },
                        _ => return "na".to_string(),
                    },
                ),
            );
        }
        ui.label(array.shape().iter().map(|x| x.to_string()).join(","));
        if !textures.contains_key(&view) {
            let mut texture = ui.ctx().load_texture(
                &artifact_id.name,
                egui::ColorImage::example(),
                egui::TextureOptions::default(),
            );
            let (min, max) = min_max_driscoll_healy(view, array);
            let color_img_info = image_from_ndarray_driscoll_healy(array, view, min, max);
            texture.set(color_img_info.image, egui::TextureOptions::default());
            textures.insert(
                view.clone(),
                ColorTextureInfo {
                    texture_handle: texture,
                    min_val: color_img_info.min_val,
                    max_val: color_img_info.max_val,
                },
            );
        }
        let pi = egui_plot::PlotImage::new(
            textures.get(&view).unwrap().texture_handle.id(),
            // texture.id(),
            egui_plot::PlotPoint::from([0.0, 0.0]),
            [2.0 * 3.14, 3.14],
        );
        // texture.set(img, egui::TextureOptions::default());
        // ui.image((texture.id(), texture.size_vec2()));
        ui.horizontal(|ui| {
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

            let color_info = textures.get(&view).unwrap();
            let t_and_color = (0..50)
                .map(|x| x as f64 / 50.0)
                .map(|t| (t, PLASMA.eval_continuous(t as f64)));
            let lines = t_and_color.tuple_windows().map(|((t1, c1), (t2, c2))| {
                let color = egui::Color32::from_rgb(c1.r, c1.g, c1.b);
                let v1 = color_info.min_val + t1 * (color_info.max_val - color_info.min_val);
                let v2 = color_info.min_val + t2 * (color_info.max_val - color_info.min_val);
                egui_plot::Line::new([[0.0, v1], [0.0, v2]].into_iter().collect_vec())
                    .color(color)
                    .width(10.0)
            });
            Plot::new((artifact_id, "colorbar"))
                .width(90.0)
                .height(plot_width / 2.0)
                .auto_bounds(egui::Vec2b::TRUE)
                .y_axis_position(egui_plot::HPlacement::Right)
                .show(ui, |plot_ui| {
                    for line in lines {
                        plot_ui.line(line)
                    }
                });
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
                    // self.dirty = true;
                    self.gui_params_sender_slot =
                        Some((self.gui_params.clone(), self.runs.runs.clone()));
                };
                if ui
                    .add(
                        egui::Slider::new(&mut self.gui_params.max_n, 500usize..=2000usize)
                            .logarithmic(true),
                    )
                    .changed()
                {
                    // self.dirty = true;
                    self.gui_params_sender_slot =
                        Some((self.gui_params.clone(), self.runs.runs.clone()));
                };
                let param_names = param_values.keys().cloned().collect_vec();
                for name in param_names.iter().sorted() {
                    if let Some(values) = self.gui_params.param_filters.get(name) {
                        if !values.is_empty() {
                            let id = ui.make_persistent_id(format!("{}_currently_filtered", name));
                            egui::collapsing_header::CollapsingState::load_with_default_open(
                                ui.ctx(),
                                id,
                                true,
                            )
                            .show_header(ui, |ui| {
                                let mut param_toggle =
                                    self.gui_params.inspect_params.contains(name);
                                ui.toggle_value(&mut param_toggle, name);
                                if !param_toggle {
                                    self.gui_params.inspect_params.remove(name);
                                } else {
                                    self.gui_params.inspect_params.insert(name.clone());
                                }
                            })
                            .body(|ui| {
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
            let id = ui.make_persistent_id(format!("{}", param_group_name));
            egui::collapsing_header::CollapsingState::load_with_default_open(
                ui.ctx(),
                id,
                !param_group_name.ends_with("_id"),
            )
            .show_header(ui, |ui| {
                if param_group_vec
                    .iter()
                    .any(|name| name.split(".").count() > depth + 1)
                {
                    ui.label(param_group_name);
                } else {
                    let mut param_toggle =
                        self.gui_params.inspect_params.contains(&param_group_name);
                    ui.toggle_value(&mut param_toggle, &param_group_name);
                    if !param_toggle {
                        self.gui_params.inspect_params.remove(&param_group_name);
                    } else {
                        self.gui_params
                            .inspect_params
                            .insert(param_group_name.clone());
                    }
                }
            })
            .body(|ui| {
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
            // egui::CollapsingHeader::new(&param_group_name)
            // .default_open(self.gui_params.param_name_filter.len() > 1)
            // .default_open(!param_group_name.ends_with("_id"))
            // .show(ui, |ui| {});
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
                    if self.table_active {
                        let stroke_color = if self.gui_params.inspect_params.contains(param_name) {
                            egui::Color32::RED
                        } else {
                            egui::Color32::TRANSPARENT
                        };
                        // if ui
                        //     .add(
                        //         egui::Button::new('\u{2795}'.to_string())
                        //             .small()
                        //             .stroke(egui::Stroke::new(1.0, stroke_color)),
                        //     )
                        //     .clicked()
                        // {
                        //     if self.gui_params.inspect_params.contains(param_name) {
                        //         self.gui_params.inspect_params.remove(param_name);
                        //     } else {
                        //         self.gui_params.inspect_params.insert(param_name.clone());
                        //     }
                        // };
                    }
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
        // if param_frame
        //     .response
        //     .interact(egui::Sense::click())
        //     .clicked()
        // {
        //     if self.gui_params.inspect_params.contains(param_name) {
        //         self.gui_params.inspect_params.remove(param_name);
        //     } else {
        //         self.gui_params
        //             .inspect_params
        //             .insert(param_name.to_string());
        //     }
        // }
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
                // self.dirty = true;
                self.update_filtered_runs();
                self.db_train_runs_sender_slot = Some(self.runs.time_filtered_runs.clone());
                self.gui_params_sender_slot =
                    Some((self.gui_params.clone(), self.runs.runs.clone()));
            }
        }
    }

    fn render_table(
        &mut self,
        ui: &mut egui::Ui,
        run_ensemble_color: &HashMap<String, egui::Color32>,
    ) {
        let param_keys = self
            .runs
            .time_filtered_runs
            .iter()
            .flat_map(|run_id| self.runs.runs.get(run_id).unwrap().params.keys())
            .unique()
            .sorted_by_key(|&param_name| {
                if self.gui_params.inspect_params.contains(param_name) {
                    (0, param_name.clone())
                } else {
                    (1, param_name.clone())
                }
            })
            .collect_vec();
        let n_params = param_keys.len();
        egui::ScrollArea::horizontal().show(ui, |ui| {
            egui_extras::TableBuilder::new(ui)
                .columns(egui_extras::Column::auto().resizable(true), n_params + 1)
                .striped(true)
                .sense(egui::Sense::click())
                .header(20.0, |mut header| {
                    header.col(|_ui| {});
                    for param_name in &param_keys {
                        let color = if self.gui_params.inspect_params.contains(*param_name) {
                            egui::Color32::GREEN
                        } else {
                            egui::Color32::WHITE
                            // ui.style().visuals.interact_cursor
                        };
                        // ui.visuals_mut().window_rounding
                        header.col(|ui| {
                            ui.heading(egui::RichText::new(*param_name).size(10.0).color(color));
                        });
                    }
                })
                .body(|mut table| {
                    for run_id in &self.runs.time_filtered_runs {
                        let run = self.runs.runs.get(run_id).unwrap();
                        let mut clipboard = None;
                        table.row(20.0, |mut row| {
                            row.col(|ui| {
                                let (rect, _) = ui.allocate_exact_size(
                                    egui::vec2(10.0, 10.0),
                                    egui::Sense::hover(),
                                );
                                ui.painter().rect(
                                    rect,
                                    0.0,
                                    run_ensemble_color.get(run_id).unwrap().clone(),
                                    egui::Stroke::new(0.0, egui::Color32::TRANSPARENT),
                                );
                            });
                            for param_key in &param_keys {
                                row.col(|ui| {
                                    if let Some(val) = run.params.get(*param_key) {
                                        if let Ok(val_f32) = val.parse::<f32>() {
                                            ui.label(format!("{:.2}", val_f32));
                                        } else {
                                            ui.add(egui::Label::new(val).wrap(false));
                                            // ui.label(val);
                                        }
                                    }
                                });
                                // ui.painter().rect
                            }
                            if row.response().clicked() {
                                clipboard = Some(run_id.clone());
                                //
                            }
                        });
                        if let Some(copied_text) = clipboard {
                            table.ui_mut().ctx().copy_text(copied_text);
                            // table.ui_mut().
                        }
                    }
                });
        });
    }

    fn render_plots(
        &mut self,
        ui: &mut egui::Ui,
        metric_names: Vec<String>,
        run_ensemble_color: &HashMap<String, egui::Color32>,
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
                        // .auto_bounds()
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
                            XAxis::Time => {
                                |grid_mark: egui_plot::GridMark,
                                 n_chars,
                                 range: &RangeInclusive<f64>| {
                                    let ts = NaiveDateTime::from_timestamp_opt(
                                        grid_mark.value as i64,
                                        0,
                                    )
                                    .unwrap();
                                    let delta = range.end() - range.start();
                                    if delta > (5 * 24 * 60 * 60) as f64 {
                                        ts.format("%m/%d").to_string()
                                    } else if delta > (5 * 60 * 60) as f64 {
                                        ts.format("%d-%Hh").to_string()
                                    } else {
                                        ts.format("%Hh:%Mm").to_string()
                                    }
                                }
                            }
                            XAxis::Batch => {
                                |grid_mark: egui_plot::GridMark,
                                 n_chars,
                                 range: &RangeInclusive<f64>| {
                                    format!("{}", grid_mark.value as i64).to_string()
                                }
                            }
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
                                        if metric.values.len() == 1 {
                                            plot_ui.points(
                                                egui_plot::Points::new(PlotPoints::from(
                                                    metric.values.clone(),
                                                ))
                                                .shape(egui_plot::MarkerShape::Circle)
                                                .radius(5.0),
                                            );
                                        }
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
        ui.collapsing("metrics", |ui| {
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
                            // self.dirty = true;
                            self.update_filtered_runs();
                            self.db_train_runs_sender_slot =
                                Some(self.runs.time_filtered_runs.clone());
                            self.gui_params_sender_slot =
                                Some((self.gui_params.clone(), self.runs.runs.clone()));
                        }
                    }
                });
            });
        });
    }

    fn render_artifact_selector(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("artifacts", |ui| {
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
                                egui::Button::new(artifact_name).selected(
                                    self.gui_params.artifact_filters.contains(artifact_name),
                                ),
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
        });
    }

    fn render_artifacts(
        &mut self,
        ui: &mut egui::Ui,
        run_ensemble_color: &HashMap<String, egui::Color32>,
    ) {
        let active_artifact_types: Vec<ArtifactType> = self
            .gui_params
            .artifact_filters
            .iter()
            .map(|artifact_name| {
                self.runs
                    .time_filtered_runs
                    .iter()
                    .map(|train_id| (train_id, self.runs.runs.get(train_id).unwrap()))
                    .filter_map(|(_train_id, run)| {
                        if let Some(artifact_id) = run.artifacts.get(artifact_name) {
                            let artifact_type = get_artifact_type(&artifact_id.name);
                            if self.artifact_handlers.contains_key(
                                self.artifact_dispatch
                                    .get(&artifact_type)
                                    .unwrap_or(&ArtifactHandlerType::Unknown),
                            ) {
                                // println!("{}", artifact_type_str);
                                // add_artifact(handler, ui, train_id, path);
                                Some(artifact_type)
                            } else {
                                None
                            }
                            // if let Some(artifact_handler) = self.
                        } else {
                            None
                        }
                    })
            })
            .flatten()
            .unique()
            .sorted()
            .collect();
        for artifact_type in active_artifact_types {
            if let Some(handler) = self.artifact_handlers.get_mut(
                self.artifact_dispatch
                    .get(&artifact_type)
                    .unwrap_or(&ArtifactHandlerType::Unknown),
            ) {
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
                    &mut self.gui_params,
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
                    // self.dirty = true;
                    self.gui_params.time_filter =
                        Some(self.runs.active_runs_time_ordered[self.gui_params.time_filter_idx].1);
                    self.update_filtered_runs();
                    self.db_train_runs_sender_slot = Some(self.runs.time_filtered_runs.clone());
                    self.gui_params_sender_slot =
                        Some((self.gui_params.clone(), self.runs.runs.clone()));
                }
            });
        }
    }

    fn update_filtered_runs(&mut self) {
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
    tx_runs: Option<&SyncSender<HashMap<String, Run>>>,
    tx_batch_status: &SyncSender<(usize, usize)>,
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
    tx_runs: Option<&SyncSender<HashMap<String, Run>>>,
    metric_table: String,
) -> Result<(), sqlx::Error> {
    if train_ids.len() > 0 {
        // let last_batch_timestamp = last_timestamp.clone();
        // let mut new_batch_timestamp = last_timestamp.clone();
        let mut chunk_size = 1000i32;
        loop {
            // profiling::scope!("iteration");
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
            // {
            //     // profiling::scope!("explain analyze");
            //     let q1 = query_string(Some("EXPLAIN ANALYZE".to_string()), q.clone());
            //     let query_time = std::time::Instant::now();
            //     let query = sqlx::query(q1.as_str())
            //         // .bind(metric_table.clone())
            //         .bind(train_ids)
            //         .bind(*last_timestamp)
            //         // .bind(last_id)
            //         .bind(chunk_size);
            //     for row in query.fetch_all(pool).await? {
            //         println!("{:?}", row.get::<String, _>(0));
            //     }
            //     println!(
            //         "explain analyze query time {}",
            //         query_time.elapsed().as_secs_f64()
            //     );
            // }
            // profiling::scope!("query");
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
            // query.fetch_all(pool).await
            // let metric_rows = tokio::runtime::Handle::current()
            // .block_on(query.fetch_all(pool))
            // .unwrap();
            let query_elapsed_time = query_time.elapsed().as_secs_f32();
            // println!("query elapsed: {}", query_elapsed_time);
            let received_rows = metric_rows.len();
            // profiling::scope!("parse");
            parse_metric_rows(metric_rows, runs, last_timestamp);
            // println!(
            //     "[db] recieved batch data from {}: {}",
            //     metric_table, received_rows
            // );
            // {
            // profiling::scope!("send");
            if let Some(tx) = tx_runs {
                tx.send(runs.clone()).expect("send failed");
                // println!("[db] sent {}", received_rows);
            }
            // }
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
    tx_runs: Option<&SyncSender<HashMap<String, Run>>>,
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
                                name: incoming_name.clone(),
                                artifact_type: get_artifact_type(&incoming_name),
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

// #[profiling::function]
fn parse_metric_rows(
    metric_rows: Vec<sqlx::postgres::PgRow>,
    runs: &mut HashMap<String, Run>,
    last_timestamp: &mut NaiveDateTime,
) {
    // profiling::scope!("sort train_id var");
    let metric_rows_sorted = metric_rows.iter().sorted_by_cached_key(|row| {
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
            // profiling::scope!("max timestamp");
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
            // profiling::scope!("processing");
            let mut orig_values: Vec<_> = rows
                .iter()
                .filter_map(|row| {
                    // profiling::scope!("filter map");
                    // println!("{:?}", row.try_get::<i32, _>("id"));
                    if let Ok(x) = row.try_get_unchecked::<f64, _>("x") {
                        // profiling::scope!("get val");
                        if let Ok(value) = row.try_get_unchecked::<f64, _>("value") {
                            if let Ok(created_at) =
                                row.try_get_unchecked::<NaiveDateTime, _>("created_at")
                            {
                                let new_value = [x, value, created_at.timestamp() as f64];
                                // profiling::scope!("exists in orig");
                                if let Some(old_orig_values) = old_orig_values {
                                    // if old_orig_values.orig_values.contains(&new_value) {
                                    if old_orig_values
                                        .orig_values
                                        .binary_search_by(|probe| probe[0].total_cmp(&x))
                                        .is_ok()
                                    {
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
                orig_values.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
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
                // profiling::scope!("extend and sort");
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
    // use tracy_client::Client;
    // let _profile_guard = Client::start();
    // console_subscriber::init();
    // Load environment variables
    // dotenv::dotenv().ok();
    let args = Args::parse();
    println!("Args: {:?}", args);
    let (tx, rx) = mpsc::sync_channel(1);
    let (tx_gui_dirty, rx_gui_dirty) = mpsc::sync_channel(100);
    let (tx_gui_recomputed, rx_gui_recomputed) = mpsc::sync_channel(100);
    let (tx_db_filters, rx_db_filters) = mpsc::sync_channel::<Vec<String>>(1);
    let rx_db_filters_am = Arc::new(std::sync::Mutex::new(rx_db_filters));
    let (tx_db_artifact, rx_db_artifact) = mpsc::sync_channel::<ArtifactTransfer>(1);
    let (tx_db_artifact_path, rx_db_artifact_path) = mpsc::sync_channel::<i32>(1);
    let (tx_batch_status, rx_batch_status) = mpsc::sync_channel::<(usize, usize)>(100);
    let rt = tokio::runtime::Runtime::new().unwrap();
    let rt_handle = rt.handle().clone();
    let _guard = rt.enter();

    std::thread::spawn(move || loop {
        // if let Ok((gui_params, mut runs)) = rx_gui_dirty.recv() {
        //     recompute(&mut runs, &gui_params);
        //     tx_gui_recomputed
        //         .send(runs)
        //         .expect("Failed to send recomputed runs.");
        // }
        let mut last_gui_params = None;
        let mut last_runs = None;

        // Drain the channel to get the last available message
        // dbg!("draining...");
        while let Ok((gui_params, runs)) = rx_gui_dirty.try_recv() {
            last_gui_params = Some(gui_params);
            last_runs = Some(runs);
        }
        // dbg!("done...");

        // Check if there was at least one message received
        if let (Some(gui_params), Some(mut runs)) = (last_gui_params, last_runs) {
            recompute(&mut runs, &gui_params);
            tx_gui_recomputed
                .send(runs)
                .expect("Failed to send recomputed runs.");
        }
    });
    rt_handle.spawn(async move {
        let database_url = env::var("DATABASE_URL").unwrap_or("postgres://postgres:postgres@localhost/equiv".to_string());
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
                        let mut new_train_ids = None;
                        while let Ok(incoming_train_ids) = rx.try_recv() {
                            new_train_ids = Some(incoming_train_ids);
                        }
                        if let Some(new_train_ids) = new_train_ids {
                            return new_train_ids;
                        }
                        // if let Ok(new_train_ids) = rx.try_recv() {
                        //     println!("new train ids");
                        //     return new_train_ids;
                        // }
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
                println!("Starting sleep");
                tokio::time::sleep(std::time::Duration::from_secs(1) - elapsed_loop_time).await;
                println!("Sleep done");
            }
        }
    });
    rt_handle.spawn(async move {
        let database_url = env::var("DATABASE_URL")
            .unwrap_or("postgres://postgres:postgres@localhost/equiv".to_string());
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(&database_url)
            .await
            .expect("Can't connect to database");
        loop {
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            if let Ok(artifact_id) = rx_db_artifact_path.try_recv() {
                handle_artifact_request(artifact_id, &pool, &tx_db_artifact).await;
            }
        }
    });
    // });

    // let options = eframe::NativeOptions {
    //     viewport: egui::ViewportBuilder::default().with_inner_size([800.0, 800.0]),
    //     ..Default::default()
    // };
    let options = eframe::NativeOptions {
        // initial_window_size: Some(egui::vec2(320.0, 240.0)),
        // initial_window_pos: Some((200., 200.).into()),
        viewport: egui::ViewportBuilder::default().with_inner_size([800.0, 800.0]),
        wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
            // supported_backends: wgpu::Backends::VULKAN,
            device_descriptor: std::sync::Arc::new(|adapter| {
                let base_limits = if adapter.get_info().backend == wgpu::Backend::Gl {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                };

                let features = wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;

                // if adapter.get_info().backend == wgpu::Backend::Vulkan {
                //     features |= wgpu::Features::SPIRV_SHADER_PASSTHROUGH;
                // }
                wgpu::DeviceDescriptor {
                    label: Some("egui wgpu device"),
                    // features: wgpu::Features::SHADER_F16 | wgpu::Features::SPIRV_SHADER_PASSTHROUGH,
                    // features: wgpu::Features::default(),
                    required_features: features,
                    required_limits: wgpu::Limits {
                        // When using a depth buffer, we have to be able to create a texture
                        // large enough for the entire surface, and we want to support 4k+ displays.
                        max_texture_dimension_2d: 8192,
                        ..base_limits
                    },
                }
            }),
            ..Default::default()
        },
        ..Default::default()
    };

    let _ = eframe::run_native(
        "Visualizer",
        options,
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            Box::<GuiRuns>::new(GuiRuns {
                runs: Default::default(),
                // dirty: true,
                db_train_runs_sender: tx_db_filters,
                db_reciever: rx,
                recomputed_reciever: rx_gui_recomputed,
                gui_params_sender: tx_gui_dirty,
                initialized: false,
                data_status: DataStatus::Waiting,
                gui_params: GuiParams {
                    max_n: 1000,
                    param_filters: HashMap::new(),
                    metric_filters: HashSet::new(),
                    inspect_params: HashSet::new(),
                    n_average: 0,
                    artifact_filters: HashSet::new(),
                    time_filter: None,
                    time_filter_idx: 0,
                    x_axis: XAxis::Batch,
                    param_name_filter: "".to_string(),
                    table_sorting: HashSet::new(),
                    npy_plot_size: 0.48,
                    render_format: cc.wgpu_render_state.as_ref().unwrap().target_format,
                },
                // texture: None,
                artifact_handlers: HashMap::from([
                    (
                        ArtifactHandlerType::SpatialNPY,
                        ArtifactHandler::NPYArtifact {
                            arrays: HashMap::new(),
                            textures: HashMap::new(),
                            views: HashMap::new(),
                            colormap_artifacts: HashSet::new(),
                            hover_lon: 0.0,
                            hover_lat: 0.0,
                        },
                    ),
                    (
                        ArtifactHandlerType::TabularNPY,
                        ArtifactHandler::NPYTabularArtifact {
                            arrays: HashMap::new(),
                            views: HashMap::new(),
                        },
                    ),
                    (
                        ArtifactHandlerType::Image,
                        ArtifactHandler::ImageArtifact {
                            // images: HashMap::new(),
                            binaries: HashMap::new(),
                        },
                    ),
                ]),
                artifact_dispatch: HashMap::from([
                    (ArtifactType::NPYHealpix, ArtifactHandlerType::SpatialNPY),
                    (
                        ArtifactType::NPYDriscollHealy,
                        ArtifactHandlerType::SpatialNPY,
                    ),
                    (ArtifactType::NPYTabular, ArtifactHandlerType::TabularNPY),
                    (ArtifactType::PngImage, ArtifactHandlerType::Image), // ("tabular".to_string(), "tabular".to_string()),
                                                                          // ("png".to_string(), "images".to_string()),
                ]),
                args,
                tx_db_artifact_path: Arc::new(Mutex::new(tx_db_artifact_path)),
                rx_db_artifact: Arc::new(Mutex::new(rx_db_artifact)),
                rx_batch_status,
                batch_status: (0, 0),
                table_active: false,
                db_train_runs_sender_slot: None,
                gui_params_sender_slot: None,
            })
        }),
    )
    .unwrap();

    Ok(())
}

async fn handle_artifact_request(
    artifact_id: i32,
    pool: &sqlx::Pool<sqlx::Postgres>,
    tx_db_artifact: &SyncSender<ArtifactTransfer>,
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
    tx: &SyncSender<HashMap<String, Run>>,
    tx_batch_status: &SyncSender<(usize, usize)>,
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
