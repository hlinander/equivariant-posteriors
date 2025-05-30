use chrono::NaiveDateTime;
use clap::Parser;
use duckdb::arrow2::legacy::utils::CustomIterTools;
use duckdb::polars::frame::DataFrame;
use duckdb::types::FromSql;
use eframe::egui_wgpu::{CallbackResources, ScreenDescriptor, WgpuSetup, WgpuSetupCreateNew};
use eframe::{egui, App};
use egui::{epaint, Stroke, TextBuffer};
use egui_code_editor::{self, highlighting::Token, CodeEditor, ColorTheme, Syntax};
use egui_file::FileDialog;
use egui_plot::{Legend, PlotPoint, PlotPoints, Points};
use egui_plot::{Line, Plot};
use itertools::Itertools;
use ndarray::{s, ArrayBase, Dim, IxDyn, IxDynImpl, OwnedRepr, SliceInfo, SliceInfoElem};
use sqlformat::{format, FormatOptions};
// use profiling::tracy_client;
use core::time;
use serde::{Deserialize, Serialize};
use sqlx::postgres::PgPoolOptions;
use sqlx::Row;
use std::borrow::{Borrow, BorrowMut};
use std::error::Error;
use std::ffi::OsStr;
use std::fs::{create_dir_all, File};
use std::hash::Hash;
use std::io::{Read, Write};
use std::num::NonZeroU64;
use std::path::Path;
use std::sync::Arc;
use std::thread::sleep;
use std::time::Instant;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tracing::{instrument, Instrument, Level};
// use wgpu::core::command::render_ffi::wgpu_render_pass_set_index_buffer;
use wgpu::util::DeviceExt;
// use sqlx::types::JsonValue
use glsl_layout::Uniform;
use std::collections::HashMap;
use std::collections::HashSet;
use std::env;
use std::ops::RangeInclusive;
use std::sync::mpsc::{self, Receiver, SyncSender};

use duckdb::polars::prelude::*;
use duckdb::{params, params_from_iter, Connection, Polars, Result};
use duckdb::{polars, DuckdbConnectionManager};
// duckdb::polars::co
use r2d2::{ManageConnection, Pool};
// use polars::prelude::DataFrame;
// use r2d2_duckdb::DuckDBConnectionManager;

pub mod np;
use colorous::PLASMA;
use ndarray_stats::QuantileExt;
use np::load_npy_bytes;

pub mod era5;

static HIDDEN_PARAMS: [&str; 3] = [
    "train_config.data.config",
    "train_config.extra.al_config.data_pool_config.subset",
    "train_config.extra.al_config.ensemble_config",
];

#[derive(Parser, Debug)]
struct Args {
    #[arg(default_value = "../../")]
    artifacts: String,
}

#[derive(Hash, Eq, PartialEq)]
struct PlotMapKey {
    // model_id: i64,
    train_id: String,
    variable: String,
    xaxis: String,
}

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
struct RunParamKey {
    train_id: String,
    name: String,
}

#[derive(PartialEq, Eq, Hash, Debug)]
struct ArtifactKey {
    train_id: String,
    name: String,
}

struct Runs2 {
    // runs: DataFrame,
    // metrics: Option<DataFrame>,
    run_params: HashMap<RunParamKey, HashSet<String>>,
    plot_map: HashMap<PlotMapKey, Vec<Vec<[f32; 2]>>>,
    artifacts: HashMap<ArtifactKey, ArtifactId>,
    artifacts_by_run: HashMap<String, Vec<ArtifactId>>,
    active_runs: Vec<String>,
}

#[derive(Default, Debug, Clone)]
enum XAxis {
    #[default]
    Batch,
    Time,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
struct RunsFilter {
    filter: HashMap<String, HashSet<String>>,
    param_name_filter: String,
    time_filter_idx: usize,
    #[serde(skip)]
    time_filter: Option<chrono::NaiveDateTime>,
}

impl RunsFilter {
    fn new() -> RunsFilter {
        Self {
            filter: HashMap::new(),
            param_name_filter: "".to_string(),
            time_filter_idx: 0,
            time_filter: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Filters {
    param_filters: Vec<RunsFilter>,
    metric_filters: HashSet<String>,
    artifact_filters: HashSet<String>,
    inspect_params: HashSet<String>,
}

#[derive(Debug, Clone)]
struct GuiParams {
    n_average: usize,
    max_n: usize,
    filters: Filters,
    // table_sorting: HashSet<String>,
    x_axis: XAxis,
    npy_plot_size: f64,
    render_format: wgpu::TextureFormat,
    param_values: HashMap<String, HashSet<String>>,
    // param_values_handle: JoinHandle<HashMap<String, HashSet<String>>>,
    filtered_values: HashMap<String, HashSet<String>>,
    hovered_run: Option<String>,
    selected_runs: Option<HashSet<String>>,
    next_param_update: std::time::Instant,
}

// impl Default for GuiParams {
// fn default() -> Self {
// Self {
// npy_plot_size: 0.48,
// ..Default::default()
// }
// }
// }

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Clone)]
struct ArtifactId {
    artifact_id: i64,
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
    min: f32,
    max: f32,
    array: ndarray::prelude::ArrayBase<
        ndarray::OwnedRepr<f32>,
        ndarray::prelude::Dim<ndarray::IxDynImpl>,
    >,
    artifact_view: NPYArtifactView,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Uniform)]
struct HPShaderUniform {
    angle1: f32,
    angle2: f32,
    min: f32,
    max: f32,
    nside: i32,
}

struct HPShaderResources {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_groups: HashMap<NPYArtifactView, wgpu::BindGroup>,
    uniform_buffers: HashMap<NPYArtifactView, wgpu::Buffer>, // hp_buf: Option<(wgpu::Buffer, usize)>,
                                                             // uniform_buf: Option<wgpu::Buffer>,
}

fn make_bind_group_layout_entry(
    binding_num: u32,
    stage: wgpu::ShaderStages,
    read_only: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: binding_num,
        visibility: stage,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
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
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        contents: byte_slice,
    });
    uniform_buf
}

fn update_uniform<T: glsl_layout::Uniform>(buffer: &wgpu::Buffer, uniform: T, queue: &wgpu::Queue) {
    let std_140_data = uniform.std140();
    let byte_slice: &[u8] = unsafe {
        core::slice::from_raw_parts(
            (&std_140_data as *const <T as Uniform>::Std140).cast::<u8>(),
            core::mem::size_of_val::<<T as Uniform>::Std140>(&std_140_data),
        )
    };
    queue.write_buffer(buffer, 0, byte_slice);
}

fn _ensure_buffer_size<'a>(
    current: &'a mut Option<(wgpu::Buffer, usize)>,
    device: &wgpu::Device,
    required_size: u64,
    usage: wgpu::BufferUsages,
) -> (&'a wgpu::Buffer, usize) {
    let buf = if current.is_none() {
        let mut buf_init_bytes = Vec::with_capacity(required_size as usize);
        buf_init_bytes.resize(required_size as usize, 0);
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            usage,
            contents: &buf_init_bytes,
        })
    } else if current.as_ref().unwrap().0.size() < required_size {
        let mut buf_init_bytes = Vec::with_capacity(required_size as usize);
        buf_init_bytes.resize(required_size as usize, 0);
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            usage,
            contents: &buf_init_bytes,
        })
    } else {
        current.take().unwrap().0
    };
    *current = Some((buf, required_size as usize));
    (&current.as_ref().unwrap().0, required_size as usize)
}

struct Test {}
impl eframe::egui_wgpu::CallbackTrait for Test {
    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        render_pass: &mut eframe::wgpu::RenderPass<'static>,
        callback_resources: &CallbackResources,
    ) {
        todo!()
    }
}

impl eframe::egui_wgpu::CallbackTrait for HPShader {
    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        render_pass: &mut eframe::wgpu::RenderPass<'static>,
        callback_resources: &CallbackResources,
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

            render_pass.set_bind_group(0, res.bind_groups.get(&self.artifact_view).unwrap(), &[]);

            render_pass.draw(0..4, 0..1);
        }
        // A command encoder executes one or many pipelines.
        // It is to WebGPU what a command buffer is to Vulkan.
    }
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        // Vec::new()
        use eframe::egui_wgpu::wgpu::{PrimitiveState, VertexState};

        if callback_resources.get::<HPShaderResources>().is_none() {
            let spectrograph_vert = load_shader(device, include_bytes!("../hpshader.vertex.spv"));
            let spectrograph_frag = load_shader(device, include_bytes!("../hpshader.fragment.spv"));
            let bindings = vec![
                make_uniform_layout_entry(0, wgpu::ShaderStages::FRAGMENT),
                make_bind_group_layout_entry(1, wgpu::ShaderStages::FRAGMENT, true),
            ];
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
                    entry_point: Some("main"),
                    buffers: &[],
                    compilation_options: Default::default(), // ..Default::default()
                },
                primitive: PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &spectrograph_frag,
                    entry_point: Some("main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.render_format,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::COLOR,
                    })],
                    compilation_options: Default::default(),
                }),
                multiview: None,
                cache: None,
            });
            callback_resources.insert(HPShaderResources {
                pipeline: render_pipeline,
                bind_groups: [].into(),
                bind_group_layout,
                // hp_buf: None,
                uniform_buffers: HashMap::new(),
                // bind_group: [((self.node, self.out_port), bind_group)].into(),
            });
            let resources = callback_resources.get_mut::<HPShaderResources>().unwrap();
            // resources.uniform_buf = Some(uniform_buf);
            self.create_or_get_bind_group(device, resources);
            // resources.hp_buf = Some((hp_buf, hp_buf_size as usize));

            // let resources = callback_resources.get_mut::<HPShaderResources>().unwrap();
        } else {
            let resources = callback_resources.get_mut::<HPShaderResources>().unwrap();
            self.create_or_get_bind_group(
                device, resources,
                // resources.uniform_buf.as_ref().unwrap(),
            );
            update_uniform(
                resources.uniform_buffers.get(&self.artifact_view).unwrap(),
                HPShaderUniform {
                    angle1: self.angle1,
                    angle2: self.angle2,
                    nside: ((self.array.shape()[self.array.shape().len() - 1] / 12) as f64).sqrt()
                        as i32,
                    min: self.min,
                    max: self.max,
                },
                queue,
            );
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

impl HPShader {
    fn create_or_get_bind_group<'a>(
        &self,
        device: &wgpu::Device,
        resources: &'a mut HPShaderResources,
        // uniform_buf: &wgpu::Buffer,
    ) -> &'a wgpu::BindGroup {
        if resources.bind_groups.contains_key(&self.artifact_view) {
            resources.bind_groups.get(&self.artifact_view).unwrap()
        } else {
            let uniform_buf = upload_uniform(
                device,
                HPShaderUniform {
                    angle1: self.angle1,
                    angle2: self.angle2,
                    nside: ((self.array.shape()[self.array.shape().len() - 1] / 12) as f64).sqrt()
                        as i32,
                    min: self.min,
                    max: self.max,
                },
            );
            resources
                .uniform_buffers
                .insert(self.artifact_view.clone(), uniform_buf);

            let hp_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: self.array.len() as u64 * core::mem::size_of::<f32>() as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: true,
            });
            let hp_buf_size = self.array.len() as u64 * core::mem::size_of::<f32>() as u64;

            let mut mapped_hp_buf = hp_buf.slice(0..).get_mapped_range_mut();
            let mapped_buffer_ptr = mapped_hp_buf.as_mut_ptr().cast::<f32>();
            if let Some(slice) = self.array.as_slice() {
                unsafe {
                    mapped_buffer_ptr.copy_from_nonoverlapping(
                        slice.as_ptr(),
                        slice.len(), // self.array.as_ptr(),
                                     // self.array.len() * core::mem::size_of::<f32>(),
                    );
                }
            }
            drop(mapped_hp_buf);
            hp_buf.unmap();

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &resources.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(
                            resources
                                .uniform_buffers
                                .get(&self.artifact_view)
                                .unwrap()
                                .as_entire_buffer_binding(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &hp_buf,
                            offset: 0,
                            size: Some(NonZeroU64::new(hp_buf_size as u64).unwrap()),
                        }),
                    },
                ],
            });
            resources
                .bind_groups
                .insert(self.artifact_view.clone(), bind_group);
            resources.bind_groups.get(&self.artifact_view).unwrap()
        }
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
    tx_path_mutex: Arc<Mutex<SyncSender<i64>>>,
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
        tx_db_artifact_path.send(artifact_id.artifact_id).unwrap();
        println!("[db] Waiting for download {:?}", artifact_id);
        let rx_db_artifact = rx_artifact_mutex.lock_owned().await;
        loop {
            let rx_res = rx_db_artifact.recv();
            match rx_res {
                Ok(artifact_binary_res) => match artifact_binary_res {
                    ArtifactTransfer::Done(artifact_binary) => {
                        return Ok(artifact_binary);
                    }
                    ArtifactTransfer::Err(artifact_binary_err) => {
                        return Err(artifact_binary_err.to_string());
                    }
                    ArtifactTransfer::Loading(downloaded, size) => {
                        tx_update
                            .send(DownloadProgressStatus { downloaded, size })
                            .unwrap();
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
    SpecedNPY,
    Image,
    Unknown,
}

enum ArtifactHandler {
    NPYArtifact {
        textures: HashMap<NPYArtifactView, ColorTextureInfo>,
        arrays: HashMap<ArtifactId, SpatialNPYArray>,
        views: HashMap<ArtifactId, NPYArtifactView>,
        _colormap_artifacts: HashSet<ArtifactId>, // container: NPYVisContainer,
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
        image_size: f32,
    },
}

#[instrument(skip_all)]
fn add_artifact(
    handler: &mut ArtifactHandler,
    artifact_id: &ArtifactId,
    _args: &Args,
    tx_path_mutex: &mut Arc<Mutex<SyncSender<i64>>>,
    rx_artifact_mutex: &Arc<Mutex<Receiver<ArtifactTransfer>>>,
) {
    let t = Instant::now();
    match handler {
        ArtifactHandler::NPYArtifact {
            textures: _,
            arrays,
            views: _,
            _colormap_artifacts: _,
            hover_lon: _,
            hover_lat: _,
        } => handle_add_npy(arrays, &artifact_id, tx_path_mutex, rx_artifact_mutex),
        // ArtifactHandler::NPYDriscollHealyArtifact { container } => handle_add_npy(
        //     &mut container.arrays,
        //     &artifact_id,
        //     tx_path_mutex,
        //     rx_artifact_mutex,
        // ),
        ArtifactHandler::ImageArtifact {
            binaries,
            image_size,
        } => match binaries.get_mut(&artifact_id) {
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
        ArtifactHandler::NPYTabularArtifact { arrays, views: _ } => {
            handle_add_npy(arrays, &artifact_id, tx_path_mutex, rx_artifact_mutex)
        }
    }
}

fn handle_add_npy(
    arrays: &mut HashMap<ArtifactId, SpatialNPYArray>,
    artifact_id: &ArtifactId,
    tx_path_mutex: &mut Arc<Mutex<SyncSender<i64>>>,
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
                        ArtifactType::NPYTabular => NPYArrayType::Tabular,
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
        // let vis_depth =
        // cdshealpix::depth(nside / (2_u32.pow(view.nside_div as u32)).clamp(1, nside));
        let vis_depth = cdshealpix::depth(2_u32.pow(view.nside_div as u32) as u32);
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
                for xu in xu - 1..=xu + 1 {
                    for yu in yu - 1..=yu + 1 {
                        let xu = xu.clamp(0, width - 1);
                        let yu = yu.clamp(0, height - 1);
                        img.pixels[yu * width + xu] = egui::Color32::from_rgb(0, 255, 0);
                    }
                }
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
    let v = *array.get(local_index.as_slice()).unwrap() as f32;
    v
}

fn slice_array(
    view: &NPYArtifactView,
    array: &ndarray::prelude::ArrayBase<
        ndarray::OwnedRepr<f32>,
        ndarray::prelude::Dim<ndarray::IxDynImpl>,
    >,
) -> ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>> {
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
    array.slice(&si).into_owned()
}

fn min_max_hp(
    view: &NPYArtifactView,
    array: &ndarray::prelude::ArrayBase<
        ndarray::OwnedRepr<f32>,
        ndarray::prelude::Dim<ndarray::IxDynImpl>,
    >,
) -> (f32, f32) {
    let aslice = slice_array(view, array);
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
    duckdb: r2d2::Pool<DuckdbConnectionManager>,
    // runs: Runs,
    runs2: Runs2,
    plot_timer: Instant,
    rx_plot_map: Receiver<(
        HashMap<ArtifactKey, ArtifactId>,
        HashMap<PlotMapKey, Vec<Vec<[f32; 2]>>>,
    )>,
    tx_active_runs: SyncSender<Vec<String>>,
    // dirty: bool,
    db_train_runs_sender: SyncSender<Vec<String>>,
    db_train_runs_sender_slot: Option<Vec<String>>,
    gui_params_sender: SyncSender<GuiParams>,
    gui_params_sender_slot: Option<GuiParams>,
    // recomputed_reciever: Receiver<HashMap<String, Run>>,
    tx_db_artifact_path: Arc<Mutex<SyncSender<i64>>>,
    rx_db_artifact: Arc<Mutex<Receiver<ArtifactTransfer>>>,
    rx_run_params: Receiver<HashMap<RunParamKey, HashSet<String>>>,
    rx_batch_status: Receiver<(usize, usize)>,
    batch_status: (usize, usize),
    initialized: bool,
    gui_params: GuiParams,
    table_active: bool,
    artifact_handlers: HashMap<ArtifactHandlerType, ArtifactHandler>,
    artifact_dispatch: HashMap<ArtifactType, ArtifactHandlerType>,
    param_values_handle: Option<JoinHandle<HashMap<String, HashSet<String>>>>,
    active_runs_handle: Option<JoinHandle<Vec<String>>>,
    filter_save_name: String,
    filter_load_dialog: Option<FileDialog>,
    filter_name_filter: String,
    table_filter: String,
    custom_plot: String,
    custom_plot_last_err: Option<String>,
    custom_plot_data: Option<DataFrame>,
    custom_plot_handle: Option<JoinHandle<Result<DataFrame>>>,
    new_filtered_values_handle: Option<JoinHandle<HashMap<String, HashSet<String>>>>,

    args: Args, // texture: Option<egui::TextureHandle>,
}

#[instrument(skip_all)]
fn get_train_ids_from_filter_duck(
    pool: &r2d2::Pool<DuckdbConnectionManager>,
    gui_params: &GuiParams,
) -> Vec<String> {
    let mut duck = pool.get().unwrap();
    let duck = duck.transaction().unwrap();

    let mut train_ids = Vec::new();
    for param_filter in &gui_params.filters.param_filters {
        let non_empty_filters = param_filter
            .filter
            .iter()
            .filter(|(_, vs)| !vs.is_empty())
            .collect_vec();
        // let p = repeat_vars(non_empty_filters.len());
        let sql_where = non_empty_filters
            .iter()
            // .zip(p)
            .map(|(param_name, values)| {
                let ored_values = values
                    .iter()
                    .map(|value| {
                        format!(
                            "(name='{}' AND value_text=?)",
                            param_name,
                            // value.escape_default()
                        )
                    })
                    .join(" OR ");
                format!("({})", ored_values)
            })
            .join(" OR ");
        let n_filters = non_empty_filters.len();
        if n_filters == 0 {
            continue;
        }
        let sql = format!(
            "
            WITH uniq_models AS (select distinct * from local.models)
            SELECT train_id, COUNT(*) FROM
                (SELECT *, name as variable, value as value_text
                FROM local.model_parameter_text
                UNION
                SELECT *, name as variable, format('{{:E}}', value) as value_text
                FROM local.model_parameter_float
                UNION
                SELECT *, name as variable, format('{{:d}}', value) as value_text
                FROM local.model_parameter_int)
            JOIN uniq_models on id=model_id WHERE {sql_where} GROUP BY (model_id, train_id) HAVING COUNT(*)={n_filters}"
        );
        let res = duck.prepare(&sql);
        if let Err(err) = &res {
            println!("{:?}", err);
        }
        if let Ok(mut stmt) = res {
            let ids: Vec<String> = stmt
                .query_map(
                    params_from_iter(
                        non_empty_filters
                            .iter()
                            .flat_map(|(_, values)| values.iter()),
                    ),
                    |row| row.get(0),
                )
                .unwrap()
                .map(|x| x.unwrap())
                .collect();
            train_ids.extend_from_slice(&ids);
        }
    }
    train_ids.into_iter().unique().collect()
}

impl eframe::App for GuiRuns {
    fn on_exit(&mut self) {
        // self.gui_params.filters
        let serialized = ron::to_string(&self.gui_params.filters).expect("Failed to serialize");

        let mut file = File::create("last_filters.ron").unwrap();
        file.write_all(serialized.as_bytes()).unwrap();
    }
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let t_update = Instant::now();

        if !self.initialized {
            ctx.set_zoom_factor(1.0);
        }
        self.handle_filtered_runs();
        self.handle_param_values();
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
        if Instant::now() > self.plot_timer {
            let s = tracing::span!(Level::TRACE, "main: send plot_timer");
            let _enter = s.enter();
            if self
                .tx_active_runs
                .try_send(self.runs2.active_runs.clone())
                .is_ok()
            {
                self.plot_timer = Instant::now() + std::time::Duration::from_secs(2);
            }
        }
        if let Ok(new_data) = self.rx_plot_map.try_recv() {
            let s = tracing::span!(Level::TRACE, "main: plot_map recv");
            let _enter = s.enter();
            (self.runs2.artifacts, self.runs2.plot_map) = new_data;
            self.runs2.artifacts_by_run = self
                .runs2
                .artifacts
                .iter()
                .sorted_by(|a, b| Ord::cmp(&a.0.train_id, &b.0.train_id))
                .group_by(|(k, v)| k.train_id.clone())
                .into_iter()
                .map(|(key, group)| {
                    (
                        key,
                        group
                            .map(|(_artifact_key, artifact_id)| artifact_id.clone())
                            .collect::<Vec<ArtifactId>>(),
                    )
                })
                .collect();
        }
        if let Ok(new_run_params) = self.rx_run_params.try_recv() {
            let s = tracing::span!(Level::TRACE, "main: update filtered runs");
            let _enter = s.enter();
            self.runs2.run_params = new_run_params;
            self.update_filtered_runs();
        }
        // while let Ok(new_runs) = self.db_reciever.try_recv() {
        //     for train_id in new_runs.keys() {
        //         if !self.runs.runs.contains_key(train_id) {
        //             let t = Instant::now();
        //             let new_active = get_train_ids_from_filter(&new_runs, &self.gui_params);
        //             self.db_train_runs_sender_slot = Some(new_active);
        //             break;
        //         }
        //     }
        //     self.gui_params_sender_slot = Some((self.gui_params.clone(), new_runs));
        //     if self.data_status == DataStatus::Waiting {
        //         self.data_status = DataStatus::FirstDataArrived;
        //     }
        // }

        if std::time::Instant::now() > self.gui_params.next_param_update {
            self.gui_params.next_param_update =
                std::time::Instant::now() + std::time::Duration::from_secs(10);
            let t = Instant::now();
            // self.gui_params.param_values = get_parameter_values_duck(&self.duckdb);
            let pool = self.duckdb.clone();
            self.param_values_handle = Some(tokio::task::spawn_blocking(move || {
                // TODO: Merge values!
                get_parameter_values_duck(&pool)
            }));
        }

        let ensemble_colors: HashMap<String, egui::Color32> = self
            .runs2
            .active_runs
            .iter()
            .map(|run_id| {
                label_from_active_inspect_params(
                    run_id.clone(),
                    &self.runs2.run_params,
                    &self.gui_params,
                )
            })
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
            .runs2
            .active_runs
            .iter()
            .map(|run_id| {
                let ensemble_id = label_from_active_inspect_params(
                    run_id.clone(),
                    &self.runs2.run_params,
                    &self.gui_params,
                );
                (run_id.clone(), *ensemble_colors.get(&ensemble_id).unwrap())
            })
            .collect();

        egui::SidePanel::left("Controls")
            .resizable(true)
            .min_width(10.0)
            .show(ctx, |ui| {
                let t = Instant::now();

                if ui.small_button("+").clicked() {
                    self.gui_params
                        .filters
                        .param_filters
                        .push(RunsFilter::new());
                }
                ui.text_edit_singleline(&mut self.filter_save_name);
                ui.text_edit_singleline(&mut self.filter_name_filter);
                if ui.small_button("save").clicked() {
                    let path = Path::new("filters/");
                    create_dir_all(path);
                    let serialized =
                        ron::to_string(&self.gui_params.filters).expect("Failed to serialize");

                    let mut file =
                        File::create(path.join(format!("{}.ron", self.filter_save_name))).unwrap();
                    file.write_all(serialized.as_bytes()).unwrap();
                }
                if ui.small_button("load").clicked() {
                    let filter = Box::new({
                        let ext = Some(OsStr::new("ron"));
                        move |path: &Path| -> bool { path.extension() == ext }
                    });
                    let mut dialog = FileDialog::open_file(Some(Path::new("filters/").into()))
                        .show_files_filter(filter);
                    dialog.open();
                    self.filter_load_dialog = Some(dialog);
                }
                if let Some(dialog) = &mut self.filter_load_dialog {
                    if dialog.show(ui.ctx()).selected() {
                        if let Some(file) = dialog.path() {
                            // self.opened_file = Some(file.to_path_buf());
                            if let Ok(mut file) = File::open(file) {
                                let mut content = String::new();
                                if let Ok(_) = file.read_to_string(&mut content) {
                                    self.gui_params.filters =
                                        ron::from_str(&content).expect("Failed to deserialize");
                                }
                            }
                        }
                    }
                }
                for param_filter in &mut self.gui_params.filters.param_filters {
                    for param_name in self.gui_params.param_values.keys() {
                        if !param_filter.filter.contains_key(param_name) {
                            param_filter
                                .filter
                                .insert(param_name.clone(), HashSet::new());
                        }
                    }
                }
                let mut temp_param_filters = self.gui_params.filters.param_filters.clone();
                let mut changed = false;
                let mut remove_idx = None;
                for i in 0..temp_param_filters.len() {
                    ui.collapsing(format!("filter {}", i), |ui| {
                        if ui.small_button("disable").clicked() {
                            remove_idx = Some(i);
                        }
                        // self.render_time_selector(ui, &mut temp_param_filters[i]);
                        self.render_params2(&mut temp_param_filters[i], ui, &run_ensemble_color);
                    });
                    changed |= temp_param_filters[i] != self.gui_params.filters.param_filters[i];
                }
                self.gui_params.filters.param_filters = temp_param_filters;
                if changed {
                    self.update_filtered_runs();
                    // for (idx, param_filter) in
                    //     self.gui_params.filters.param_filters.iter().enumerate()
                    // {
                    //     for (k, v) in param_filter.filter.iter() {
                    //         if !v.is_empty() {
                    //         }
                    //     }
                    // }
                    self.db_train_runs_sender_slot = Some(self.runs2.active_runs.clone());
                }
                if let Some(remove_idx) = remove_idx {
                    self.gui_params.filters.param_filters.remove(remove_idx);
                    self.update_filtered_runs();
                }
            });
        let t = Instant::now();

        egui::SidePanel::right("Metrics")
            .resizable(true)
            .default_width(300.0)
            .width_range(100.0..=500.0)
            .show(ctx, |ui| {
                let metric_names = self
                    .runs2
                    .plot_map
                    .keys()
                    .map(|k| k.variable.clone())
                    .unique()
                    .sorted()
                    .collect();
                self.render_metrics(ui, &metric_names);
                ui.separator();
                self.render_artifact_selector(ui);
                ui.separator();
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            // ui.style_mut().debug.show_expand_height = true;
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    if ui.button("Time").clicked() {
                        self.gui_params.x_axis = XAxis::Time;
                    }
                    if ui.button("Batch").clicked() {
                        self.gui_params.x_axis = XAxis::Batch;
                    }
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
                .max_width(f32::INFINITY)
                .id_source("central_space")
                .show(ui, |ui| {
                    ui.allocate_space(egui::Vec2::new(ui.available_width(), 0.0));
                    let collapsing = ui.collapsing("Tabular", |ui| {
                        self.render_table(
                            ui,
                            &run_ensemble_color,
                            self.runs2.active_runs.clone(),
                            false,
                        );
                    });
                    self.table_active = collapsing.fully_open();
                    if let Some(selected_runs) = &self.gui_params.selected_runs {
                        self.render_table(
                            ui,
                            &run_ensemble_color,
                            selected_runs.clone().into_iter().collect(),
                            true,
                        );
                    }
                    let t = Instant::now();
                    self.render_artifacts(ui, &run_ensemble_color);

                    ui.collapsing("Custom plot", |ui| {
                        self.render_custom_plot(ui);
                    });
                    self.render_plots2(ui, &run_ensemble_color);
                });
        });
        self.initialized = true;
        ctx.request_repaint();
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Ord, PartialOrd, Debug)]
enum ArtifactType {
    PngImage,
    NPYHealpix,
    NPYDriscollHealy,
    NPYTabular,
    NPYSpeced,
    Unknown,
}

fn get_artifact_type(path: &String) -> ArtifactType {
    match path.split(".").last().unwrap_or("unknown") {
        "npy" => ArtifactType::NPYHealpix,
        "npydh" => ArtifactType::NPYDriscollHealy,
        "tabular" => ArtifactType::NPYTabular,
        "npyspec" => ArtifactType::NPYSpeced,
        "png" => ArtifactType::PngImage,
        "clifford_rope_test" => ArtifactType::PngImage,
        _ => ArtifactType::Unknown,
    }
}

fn label_from_active_inspect_params(
    train_id: String,
    run_params: &HashMap<RunParamKey, HashSet<String>>,
    gui_params: &GuiParams,
) -> String {
    let label = if gui_params.filters.inspect_params.is_empty() {
        run_params
            // .get(&(train_id, "ensemble_id".into()))
            .get(&RunParamKey {
                train_id,
                name: "ensemble_id".into(),
            })
            .unwrap_or(&["none000".to_string()].into_iter().collect())
            .clone()
            .iter()
            .sorted()
            .map(|v| v[0..6].to_string())
            .join(",")
        // .to_string()
    } else {
        let empty = "".to_string();
        gui_params
            .filters
            .inspect_params
            .iter()
            .sorted()
            .map(|param| {
                format!(
                    "{}:{}",
                    param.split(".").last().unwrap_or(param),
                    run_params
                        .get(&RunParamKey {
                            train_id: train_id.clone(),
                            name: param.clone()
                        })
                        // .get(&(train_id.clone(), param.clone()))
                        .unwrap_or(&[empty.clone()].into_iter().collect())
                        .iter()
                        .sorted()
                        .join(",")
                )
            })
            .join(", ")
    };
    label
}
// fn label_from_active_inspect_params2(runs: &DataFrame, train_id: String, gui_params: &GuiParams) -> String {
//     let label = if gui_params.inspect_params.is_empty() {
//         // run.params.get("ensemble_id").unwrap().clone()
//         // runs.filter()
//         runs.column("name").unwrap().equal("ensemble_id");
//         runs.column("train_id").unwrap().utf8().unwrap().equal(train_id);
//         String::new()
//         // runs.
//     } else {
//         let empty = "".to_string();
//         gui_params
//             .inspect_params
//             .iter()
//             .sorted()
//             .map(|param| {
//                 format!(
//                     "{}:{}",
//                     param.split(".").last().unwrap_or(param),
//                     run.params.get(param).unwrap_or(&empty)
//                 )
//             })
//             .join(", ")
//     };
//     label
// }
fn show_artifacts(
    ui: &mut egui::Ui,
    handler: &mut ArtifactHandler,
    gui_params: &mut GuiParams,
    // runs: &HashMap<String, Run>,
    filtered_runs: &Vec<String>,
    run_ensemble_color: &HashMap<String, egui::Color32>,
    run_params: &HashMap<RunParamKey, HashSet<String>>,
) {
    match handler {
        ArtifactHandler::NPYArtifact {
            textures,
            arrays,
            views,
            _colormap_artifacts: _,
            hover_lon,
            hover_lat,
        } => {
            // let texture = texture.get_or_insert_with(|| {});
            let mut to_remove = Vec::new();
            let npy_axis_id = ui.id().with("npy_axis");
            let available_artifact_names: Vec<&String> = arrays.keys().map(|id| &id.name).collect();
            // let mut to_be_reloaded = None;
            {
                for (_artifact_name, filtered_arrays) in gui_params
                    .filters
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
                                ui.horizontal_wrapped(|ui| {
                                    ui.label(egui::RichText::new(artifact_name).size(20.0));
                                    // ui.end_row();
                                    // ui.allocate_space(egui::Vec2::new(ui.available_width(), 0.0));
                                    for (artifact_id, array) in array_group {
                                        ui.end_row();
                                        if ui.button("reload").clicked() {
                                            to_remove.push(artifact_id.clone());
                                        }
                                        // ui.end_row();
                                        // ui.allocate_space(egui::Vec2::new(ui.available_width(), 0.0));
                                        match &array.array {
                                            NPYArray::Loading(binary_artifact) => {
                                                render_artifact_download_progress(
                                                    &binary_artifact,
                                                    ui,
                                                );
                                            }
                                            NPYArray::Loaded(npyarray) => {
                                                // ui.allocate_ui()
                                                // if ui.button("reload").clicked() {
                                                // to_be_reloaded = Some(artifact_id);
                                                // array.array
                                                // }
                                                ui.allocate_ui(
                                                    egui::Vec2::from([plot_width, plot_height]),
                                                    |ui| match array.array_type {
                                                        NPYArrayType::HealPix => {
                                                            render_npy_artifact_hp(
                                                                ui,
                                                                // runs,
                                                                artifact_id,
                                                                gui_params,
                                                                run_ensemble_color,
                                                                run_params,
                                                                views,
                                                                &npyarray,
                                                                textures,
                                                                plot_width,
                                                                npy_axis_id,
                                                                hover_lon,
                                                                hover_lat,
                                                            );
                                                            // ui.painter().rect(
                                                            //     ui.cursor(),
                                                            //     0.0,
                                                            //     egui::Color32::RED,
                                                            //     (0.0, egui::Color32::TRANSPARENT),
                                                            // );
                                                        }
                                                        NPYArrayType::DriscollHealy => {
                                                            render_npy_artifact_driscoll_healy(
                                                                ui,
                                                                // runs,
                                                                artifact_id,
                                                                gui_params,
                                                                run_ensemble_color,
                                                                run_params,
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
        ArtifactHandler::ImageArtifact {
            binaries,
            image_size,
        } => {
            // egui::Resize
            ui.add(egui::Slider::new(image_size, 0.0..=1.0).text("image size"));
            let max_size = egui::Vec2::new(
                ui.available_width() * *image_size,
                ui.available_height() * *image_size,
            );
            let available_artifact_names: Vec<&String> =
                binaries.keys().map(|id| &id.name).collect();
            ui.horizontal_wrapped(|ui| {
                // ui.painter().rect(
                //     egui::Rect::from_min_size(
                //         ui.cursor().min,
                //         [ui.available_size_before_wrap().x, 10.0].into(),
                //     ),
                //     0.0,
                //     egui::Color32::RED,
                //     (0.0, egui::Color32::TRANSPARENT),
                // );
                // ui.set_max_width(ui.available_width());
                // ui.allocate_space(egui::Vec2::new(ui.available_width(), 0.0));
                for (_artifact_name, filtered_arrays) in gui_params
                    .filters
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
                        // let (r, _) = ui.allocate_exact_size(max_size, egui::Sense::hover());
                        // ui.painter().rect(
                        // r,
                        // 0.0,
                        // egui::Color32::RED,
                        // egui::Stroke::new(0.0, egui::Color32::TRANSPARENT),
                        // );
                        // continue;
                        ui.allocate_ui(max_size, |ui| {
                            ui.push_id(artifact_id, |ui| {
                                ui.vertical(|ui| {
                                    let label = label_from_active_inspect_params(
                                        artifact_id.train_id.clone(),
                                        run_params,
                                        &gui_params,
                                    );
                                    ui.colored_label(
                                        run_ensemble_color
                                            .get(&artifact_id.train_id)
                                            .unwrap()
                                            .clone(),
                                        egui::RichText::new(format!(
                                            "{}: {}",
                                            artifact_id.name, label
                                        ))
                                        .size(14.0 * *image_size),
                                    );
                                    // ui.allocate_space(max_size);
                                    // if let Some(binary) = binaries.get(&artifact_id) {
                                    match binary_artifact {
                                        BinaryArtifact::Loaded(binary_data) => {
                                            // let mut file = File::create("test.png").unwrap();
                                            // file.write_all(&binary_data).unwrap();
                                            // file.flush();
                                            // panic!();
                                            let span =
                                                tracing::span!(tracing::Level::INFO, "add image");
                                            let _guard = span.enter();
                                            ui.add(
                                                egui::Image::from_bytes(
                                                    format!(
                                                        "bytes://{}_{}.png",
                                                        artifact_id.train_id, artifact_id.name
                                                    ),
                                                    binary_data.clone(),
                                                )
                                                .fit_to_exact_size(max_size * 0.9),
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
            for (_artifact_name, filtered_arrays) in gui_params
                .filters
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
                for (artifact_id, array) in filtered_arrays {
                    match &array.array {
                        NPYArray::Loading(binary_artifact) => {
                            render_artifact_download_progress(&binary_artifact, ui);
                        }
                        NPYArray::Loaded(array) => {
                            // ui.allocate_ui()
                            ui.allocate_ui(
                                egui::Vec2::from([plot_width, plot_height + 200.0]),
                                |ui| {
                                    render_npy_artifact_tabular(
                                        ui,
                                        // runs,
                                        artifact_id,
                                        gui_params,
                                        run_ensemble_color,
                                        run_params,
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
        BinaryArtifact::Loaded(_data) => {
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
    artifact_id: &ArtifactId,
    gui_params: &GuiParams,
    run_ensemble_color: &HashMap<String, egui::Color32>,
    run_params: &HashMap<RunParamKey, HashSet<String>>,
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
            label_from_active_inspect_params(artifact_id.train_id.clone(), run_params, &gui_params);
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
                ui.add(
                    egui::Slider::new(&mut view.nside_div, 0..=16)
                        .custom_formatter(|index_f, _| format!("{}", 2_u32.pow(index_f as u32))),
                );
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
                let min = textures.get(&view).unwrap().min_val as f32;
                let max = textures.get(&view).unwrap().max_val as f32;
                render_hp_shader(
                    ui,
                    view,
                    array,
                    gui_params,
                    *hover_lon,
                    *hover_lat,
                    min,
                    max,
                    plot_width / 2.0,
                    plot_width / 2.0,
                );
                Plot::new(artifact_id)
                    .width(plot_width)
                    .height(plot_width / 2.0)
                    .data_aspect(1.0)
                    .view_aspect(1.0)
                    .show_grid(false)
                    .link_axis(npy_axis_id, [true, true])
                    .link_cursor(npy_axis_id, [true, true])
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
                let lines = t_and_color.tuple_windows().map(|((t1, c1), (t2, _c2))| {
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
            // ui.painter().debug_rect(ui., , )
            // ui.debug_paint_cursor();
        });
    });
}

fn render_hp_shader(
    ui: &mut egui::Ui,
    view: &mut NPYArtifactView,
    array: &ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
    gui_params: &GuiParams,
    lon: f64,
    lat: f64,
    min: f32,
    max: f32,
    width: f32,
    height: f32,
) {
    let (resp, painter) = ui.allocate_painter(
        egui::Vec2::new(width, height),
        egui::Sense::focusable_noninteractive(),
    );
    // let angle1 = if let Some(pos) = ui.ctx().pointer_latest_pos() {
    //     pos.x / 200.0
    // } else {
    //     0.0
    // };
    // let angle2 = if let Some(pos) = ui.ctx().pointer_latest_pos() {
    //     pos.y / 200.0
    // } else {
    //     0.0
    // };
    let subarray = slice_array(view, array);
    // dbg!(&view);
    // dbg!(&subarray);
    painter.add(egui::Shape::Callback(
        eframe::egui_wgpu::Callback::new_paint_callback(
            resp.rect,
            HPShader {
                render_format: gui_params.render_format,
                angle1: lon as f32,
                angle2: lat as f32,
                min,
                max, // min: min_max_hp(, ),
                array: subarray,
                artifact_view: view.clone(), // array: array.clone(),
                                             // max: 0.0, // node: *node_idx,
                                             // out_port: output_id,
            },
        ),
    ));
}
fn render_npy_artifact_driscoll_healy(
    ui: &mut egui::Ui,
    // runs: &HashMap<String, Run>,
    artifact_id: &ArtifactId,
    gui_params: &GuiParams,
    run_ensemble_color: &HashMap<String, egui::Color32>,
    run_params: &HashMap<RunParamKey, HashSet<String>>,
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
            label_from_active_inspect_params(artifact_id.train_id.clone(), run_params, &gui_params);
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
                .link_axis(npy_axis_id, [true, true])
                .link_cursor(npy_axis_id, [true, true])
                .show(ui, |plot_ui| {
                    plot_ui.image(pi);
                });

            let color_info = textures.get(&view).unwrap();
            let t_and_color = (0..50)
                .map(|x| x as f64 / 50.0)
                .map(|t| (t, PLASMA.eval_continuous(t as f64)));
            let lines = t_and_color.tuple_windows().map(|((t1, c1), (t2, _c2))| {
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
    // runs: &HashMap<String, Run>,
    artifact_id: &ArtifactId,
    gui_params: &GuiParams,
    run_ensemble_color: &HashMap<String, egui::Color32>,
    run_params: &HashMap<RunParamKey, HashSet<String>>,
    views: &mut HashMap<ArtifactId, NPYTabularArtifactView>,
    array: &ndarray::prelude::ArrayBase<
        ndarray::OwnedRepr<f32>,
        ndarray::prelude::Dim<ndarray::IxDynImpl>,
    >,
    plot_width: f32,
) {
    ui.vertical(|ui| {
        let label =
            label_from_active_inspect_params(artifact_id.train_id.clone(), run_params, &gui_params);
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
        let _n_rows = array.shape()[0];
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
        let _plot = Plot::new(artifact_id)
            .width(plot_width)
            .height(plot_width / 2.0)
            .auto_bounds(egui::Vec2b::TRUE)
            // .auto_bounds_x()
            // .auto_bounds_y()
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
                // let xy = xs
                //     .to_slice()
                //     .unwrap()
                //     .iter()
                //     .zip(ys.to_slice().unwrap())
                //     .map(|x| [*x.0 as f64, *x.1 as f64])
                //     .collect_vec();
                // let v = xy.iter().next();
                // xs.fold_axis(, , )
                plot_ui.points(Points::new(xy));
                // plot_ui.image(pi);
            });
    });
}
enum Tree {
    Node {
        cat: String,
        rest: String,
        path: String,
        full_cat: String,
    },
    Leaf(String, String),
}

fn contains_str(list_tree: &Vec<Tree>, substr: &str) -> bool {
    list_tree.iter().any(|x| match x {
        Tree::Node {
            cat: _,
            rest: _,
            path,
            full_cat: _,
        } => path.contains(substr),
        Tree::Leaf(_, path) => path.contains(substr),
    })
}

#[derive(PartialEq)]
enum Group {
    Leaf,
    Category(String, String),
}

#[instrument(skip_all)]
fn update_plot_map(
    pool: &r2d2::Pool<DuckdbConnectionManager>,
    active_runs: &Vec<String>,
) -> HashMap<PlotMapKey, Vec<Vec<[f32; 2]>>> {
    // self.runs2.plot_map.clear();
    let mut plot_map = HashMap::new();
    if active_runs.len() == 0 {
        return HashMap::new();
    }
    // let duck = pool.get().unwrap();
    // let metrics_sql = format!(
    //     "
    //         select distinct variable
    //         from local.metrics
    //         where xaxis='batch' AND train_id IN ({})",
    //     repeat_vars(active_runs.len())
    // );
    // // let mut stmt = duck.prepare(&metrics_sql).unwrap();
    // let rows = stmt
    //     .query_map(duckdb::params_from_iter(active_runs.iter()), |row| {
    //         row.get(0)
    //     })
    //     .unwrap();
    // let metric_names: Vec<_> = rows.map(|x| x.unwrap()).collect();

    let mut df = get_metrics_duck(pool, active_runs);
    // df.writejkj
    // polars::io::csv::CsvWriter::new()

    if df.height() == 0 {
        return HashMap::new();
    }
    // for (name, data) in df.group_by(["train_id", "variable"]).unwrap() {}
    let model_ids = df.column("model_id").unwrap().rechunk();
    let mut model_ids = model_ids.as_materialized_series().iter();
    let train_ids = df.column("train_id").unwrap().rechunk();
    let mut train_ids = train_ids.as_materialized_series().iter();
    let variables = df.column("name").unwrap().rechunk();
    let mut variables = variables.as_materialized_series().iter();
    // let xaxis = df.column("xaxis").unwrap().rechunk();
    // let mut xaxis = xaxis.as_materialized_series().iter();
    let xs = df.column("xs").unwrap().rechunk();
    let mut xs = xs.as_materialized_series().iter();
    let values = df.column("values").unwrap().rechunk();
    // let y_mean = values.mean().unwrap();
    // let y_std = match values.std_as_series(0).get(0).unwrap() {
    // AnyValue::Float32(v) => v as f64,
    // AnyValue::Float64(v) => v,
    // _ => panic!(),
    // };
    let mut values = values.as_materialized_series().iter();
    // let mut iters = df
    //     .columns(["train_id", "variable", "x", "value"])
    //     .unwrap()
    //     .iter()
    //     .map(|c| c.iter())
    //     .collect::<Vec<_>>();
    // let mut last_train_id = String::new();
    for row in 0..df.height() {
        let mid = model_ids.next().unwrap();
        let tid = train_ids.next().unwrap();
        let train_id = tid.get_str().unwrap();
        let var = variables.next().unwrap();
        let variable = var.get_str().unwrap();
        // let xaxis = xaxis.next().unwrap();
        // let xaxis = xaxis.get_str().unwrap();
        let x = xs.next().unwrap();
        let x = match x {
            AnyValue::List(sx) => sx
                .f64()
                .unwrap()
                .to_vec()
                .into_iter()
                .flatten()
                .map(|x| x as f32)
                .collect::<Vec<f32>>(),
            _ => {
                panic!()
            }
        };
        let y = values.next().unwrap();
        let y = match y {
            AnyValue::List(sy) => sy
                .f64()
                .unwrap()
                .to_vec()
                .into_iter()
                .flatten()
                .map(|x| x.log10() as f32)
                .collect::<Vec<f32>>(),
            _ => {
                panic!()
            }
        };
        // for (idx, w) in x.windows(2).enumerate() {
        //     if w[1] < w[0] {
        //             "{}: {}, {} < {} [{idx} ({})]",
        //             train_id,
        //             variable,
        //             w[1],
        //             w[0],
        //             x.len()
        //         );
        //         // panic!();
        //         let output_file: File =
        //             File::create("out.json").expect("Failed to create an output file.");

        //         let mut writer: polars::io::json::JsonWriter<File> = JsonWriter::new(output_file);

        //         writer
        //             .finish(&mut df)
        //             .expect("Failed to write the CSV file.");
        //         panic!();
        //     }
        // }
        let xy = x
            .into_iter()
            .zip(y)
            .filter_map(|(x, y)| {
                // let z = (y as f64 - y_mean) / y_std;
                if y.abs() < 10000.0 {
                    Some([x, y])
                } else {
                    None
                }
            })
            .collect_vec();
        let AnyValue::Int64(mid) = mid else { panic!() };
        plot_map
            .entry(PlotMapKey {
                train_id: train_id.into(),
                variable: variable.into(),
                xaxis: "batch".to_string(), // xaxis: xaxis.into(),
            })
            .or_insert(vec![])
            .push(xy);
        //         plot_map.insert(
        // ,
        //             // (
        //             //     variable.to_string(),
        //             //     train_id.to_string(),
        //             //     xaxis.to_string(),
        //             // ),
        //             xy,
        //         );
    }
    // for x in df.group_by(["train_id", "variable"]).unwrap() {}
    return plot_map;
    // return HashMap::new();

    let sql = format!(
        "
WITH range_info AS (
    SELECT 
        MIN(x) AS min_x, 
        MAX(x) AS max_x 
    FROM local.metrics
    WHERE train_id = $2 AND variable = $1
),
bucket_size AS (
    SELECT 
        (max_x - min_x) / 10 AS size 
    FROM range_info
)
SELECT 
    t.train_id as train_id, 
    t.variable as variable,
    FLOOR(x / bs.size) AS bucket, 
    AVG(value) AS value,
    AVG(x) AS x
FROM local.metrics t
JOIN bucket_size bs 
ON t.train_id = $2 AND t.variable = $1
WHERE t.train_id = $2 AND t.variable = $1
GROUP BY train_id, variable, bucket
ORDER BY x;
        "
    );
    // let sql = format!("select x, value from local.metrics where xaxis='batch' AND variable=$1 AND train_id=$2 ORDER BY x");
    // let mut stmt = duck.prepare(&sql).expect("plotmap");
    // for metric_name in metric_names.iter().sorted() {
    //     for train_id in active_runs.iter().sorted() {
    //         let polars = stmt.query_polars([metric_name, train_id]).expect("plotmap");
    //         if let Some(df) = polars.reduce(|acc, e| acc.vstack(&e).unwrap()) {
    //             // df.column("x").unwrap().iter()
    //             let n = df.shape().0;
    //             let window_size = 1.max(n / 1000) as i64;
    //             // let df = df
    //             //     .lazy()
    //             //     .with_columns([
    //             //         col("x").rolling_mean(RollingOptions {
    //             //             window_size: Duration::new(window_size),
    //             //             min_periods: 1,
    //             //             ..Default::default()
    //             //         }),
    //             //         col("value").rolling_mean(RollingOptions {
    //             //             window_size: Duration::new(window_size),
    //             //             min_periods: 1,
    //             //             ..Default::default()
    //             //         }),
    //             //     ])
    //             //     .collect()
    //             //     .unwrap();
    //             let x = df.column("x").unwrap().f32().unwrap();
    //             let x = x.to_vec().into_iter().step_by(window_size as usize);
    //             let y = df.column("value").unwrap().f32().unwrap().rechunk();
    //             let y = y.to_vec().into_iter().step_by(window_size as usize);
    //             let xy = x
    //                 .zip(y)
    //                 .map(|(x, y)| [x.unwrap_or(0.0) as f64, y.unwrap_or(0.0) as f64])
    //                 .collect_vec();
    //             plot_map.insert((metric_name.clone(), train_id.clone()), xy);
    //         }
    //     }
    // }
    // plot_map
}
// fn update_plot_map2(
//     pool: &r2d2::Pool<DuckdbConnectionManager>,
//     active_runs: &Vec<String>,
// ) -> HashMap<(String, String), Vec<[f64; 2]>> {
//     // self.runs2.plot_map.clear();
//     let mut plot_map = HashMap::new();
//     if active_runs.len() == 0 {
//         return HashMap::new();
//     }

//     let metrics = get_metrics_duck(pool, active_runs);
//     let metric_names = metrics
//         .column("variable")
//         .unwrap()
//         .sort(false)
//         .unique_stable()
//         .unwrap();
//     for metric_name in metric_names.iter() {
//         for train_id in active_runs.iter().sorted() {
//             // let df =
//             let x = df.column("x").unwrap().f32().unwrap();
//             let x = x.to_vec().into_iter().step_by(window_size as usize);
//             let y = df.column("value").unwrap().f32().unwrap().rechunk();
//             let y = y.to_vec().into_iter().step_by(window_size as usize);
//             let xy = x
//                 .zip(y)
//                 .map(|(x, y)| [x.unwrap_or(0.0) as f64, y.unwrap_or(0.0) as f64])
//                 .collect_vec();
//             plot_map.insert((metric_name.clone(), train_id.clone()), xy);
//         }
//     }
//     plot_map
// }
impl GuiRuns {
    fn render_table(
        &mut self,
        ui: &mut egui::Ui,
        run_ensemble_color: &HashMap<String, egui::Color32>,
        runs: Vec<String>,
        show_diff: bool,
    ) {
        ui.text_edit_singleline(&mut self.table_filter);
        let param_keys = self
            .runs2
            .run_params
            .keys()
            .map(|key| key.name.clone())
            .filter(|name| {
                !self
                    .gui_params
                    .filtered_values
                    .get(name)
                    .unwrap_or(&HashSet::new())
                    .is_empty()
                    && (name.contains(&self.table_filter) || self.table_filter.is_empty())
                    && !runs.iter().all(|train_id| {
                        self.runs2
                            .run_params
                            .get(&RunParamKey {
                                train_id: train_id.clone(),
                                name: name.clone(),
                            })
                            .is_none()
                    })
            })
            .unique()
            .sorted_by_key(|param_name| {
                if self.gui_params.filters.inspect_params.contains(param_name) {
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
                        let color = if self.gui_params.filters.inspect_params.contains(param_name) {
                            egui::Color32::GREEN
                        } else {
                            egui::Color32::BLUE
                            // ui.style().visuals.text_color()
                            // ui.style().visuals.interact_cursor
                        };
                        // ui.visuals_mut().window_rounding
                        header.col(|ui| {
                            ui.heading(egui::RichText::new(param_name).size(10.0).color(color));
                        });
                    }
                })
                .body(|mut table| {
                    for run_id in runs.iter().sorted() {
                        // let run = self.runs.runs.get(run_id).unwrap();
                        let mut clipboard = None;
                        table.row(20.0, |mut row| {
                            let hovered = if let Some(hovered) = &self.gui_params.hovered_run {
                                *hovered == *run_id
                            } else {
                                false
                            };
                            let selected =
                                if let Some(selected_runs) = &self.gui_params.selected_runs {
                                    // *selected == *run_id
                                    selected_runs.contains(run_id)
                                } else {
                                    false
                                };
                            row.set_selected(hovered || selected);
                            row.col(|ui| {
                                let (rect, _) = ui.allocate_exact_size(
                                    egui::vec2(10.0, 10.0),
                                    egui::Sense::hover(),
                                );
                                ui.painter().rect(
                                    rect,
                                    0.0,
                                    run_ensemble_color
                                        .get(run_id)
                                        .unwrap_or(&egui::Color32::WHITE)
                                        .clone(),
                                    egui::Stroke::new(0.0, egui::Color32::TRANSPARENT),
                                    egui::StrokeKind::Middle,
                                );
                            });
                            for param_key in &param_keys {
                                row.col(|ui| {
                                    if let Some(val) = self.runs2.run_params.get(&RunParamKey {
                                        train_id: run_id.clone(),
                                        name: param_key.clone(),
                                    })
                                    // .get(&(run_id.clone(), (*param_key).clone()))
                                    {
                                        if show_diff
                                            && !runs
                                                .iter()
                                                .map(|train_id| {
                                                    self.runs2.run_params.get(&RunParamKey {
                                                        train_id: train_id.clone(),
                                                        name: param_key.clone(),
                                                    })
                                                })
                                                .all_equal()
                                        {
                                            let (_, rect) = ui.allocate_space((5.0, 5.0).into());
                                            ui.painter().rect(
                                                rect,
                                                0.0,
                                                egui::Color32::RED,
                                                egui::Stroke::NONE,
                                                egui::StrokeKind::Middle,
                                            );
                                        }
                                        // if let Ok(val_f32) = val.parse::<f32>() {
                                        // ui.label(format!("{:.2}", val_f32));
                                        // } else {
                                        ui.add(egui::Label::new(val.iter().join(",")).truncate());
                                        // ui.label(val);
                                        // }
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

    fn render_params2(
        &mut self,
        param_filter: &mut RunsFilter,
        ui: &mut egui::Ui,
        run_ensemble_color: &HashMap<String, egui::Color32>,
    ) {
        let v = self
            .gui_params
            .param_values
            .keys()
            .unique()
            .sorted()
            .collect_vec();

        let tree = v
            .into_iter()
            .map(|x| match x.split_once(".") {
                Some((cat, rest)) => Tree::Node {
                    cat: cat.to_string(),
                    rest: rest.to_string(),
                    path: x.clone(),
                    full_cat: cat.to_string(),
                },
                None => Tree::Leaf(x.clone(), x.clone()),
            })
            .collect_vec();

        let mut diff_counts = HashMap::new();
        let counts: Vec<_> = self
            .gui_params
            .filtered_values
            .iter()
            .map(|(k, v)| (k.as_str(), v.len()))
            .collect();
        let mut counts: Vec<_> = counts.into_iter().sorted_by_key(|(k, v)| *k).collect();
        diff_counts.extend(counts.clone().into_iter());
        let mut num_groups = counts.len();
        while num_groups > 1 {
            let groups = counts.into_iter().group_by(|(param_name, count)| {
                if let Some((left, right)) = param_name.rsplit_once(".") {
                    left
                } else {
                    param_name
                }
            });
            let level = groups
                .into_iter()
                .map(|(param_name, group)| {
                    (
                        param_name,
                        group.into_iter().map(|(_, count)| count).max().unwrap_or(0),
                    )
                })
                .collect_vec();
            diff_counts.extend(level.clone().into_iter());
            num_groups = level.len();
            if !level.iter().any(|x| x.0.contains(".")) {
                num_groups = 1;
            }
            counts = level;
        }

        egui::ScrollArea::vertical().show(ui, |ui| {
            render_param_tree(
                &mut self.gui_params.filters.inspect_params,
                param_filter,
                &self.gui_params.filtered_values,
                // &runs,
                &self.runs2.run_params,
                tree,
                ui,
                run_ensemble_color,
                &diff_counts,
                &self.filter_name_filter,
            );
        });
    }

    fn render_custom_plot(&mut self, ui: &mut egui::Ui) {
        // ui.text_edit_multiline(&mut self.custom_plot);
        let res = CodeEditor::default()
            .id_source("sql_code")
            .with_rows(15)
            .with_fontsize(14.0)
            .with_theme(ColorTheme::SONOKAI)
            .with_syntax(Syntax::sql())
            .with_numlines(true)
            .show(ui, &mut self.custom_plot);
        if res.response.has_focus() {
            if ui.input(|input| input.key_pressed(egui::Key::S) && input.modifiers.command) {
                self.custom_plot = sqlformat::format(
                    &self.custom_plot,
                    &sqlformat::QueryParams::None,
                    FormatOptions::default(),
                );
            }
        }

        if ui.button("format").clicked() {
            self.custom_plot = sqlformat::format(
                &self.custom_plot,
                &sqlformat::QueryParams::None,
                FormatOptions::default(),
            );
        }
        if ui.button("run").clicked()
            || (res.response.has_focus()
                && ui.input(|input| input.key_pressed(egui::Key::Enter) && input.modifiers.command))
        {
            let conn = self.duckdb.get().unwrap();
            let query = self.custom_plot.clone();
            self.custom_plot_handle = Some(tokio::task::spawn_blocking(move || {
                conn.execute_batch("DROP TABLE IF EXISTS params;  CREATE TABLE params AS pivot local.runs on variable using any_value(value_text) group by train_id").unwrap();
                match conn.prepare(&query) {
                    Ok(mut stmt) => {
                        let polars = stmt.query_polars([]).unwrap();
                        let large_df = polars.reduce(|acc, e| acc.vstack(&e).unwrap());
                        let large_df = large_df.unwrap_or(DataFrame::empty());
                        Ok(large_df)
                    }
                    Err(err) => {
                        Err(err)
                        // self.custom_plot_last_err = Some(err.to_string());
                        // ui.label(err.to_string());
                    }
                }
            }));
        }
        if let Some(handle) = self.custom_plot_handle.take() {
            if handle.is_finished() {
                let large_df = tokio::runtime::Handle::current().block_on(handle).unwrap();
                match large_df {
                    Ok(large_df) => {
                        self.custom_plot_data = Some(large_df);
                        self.custom_plot_last_err = None;
                    }
                    Err(err) => {
                        self.custom_plot_last_err = Some(err.to_string());
                    }
                }
            } else {
                ui.spinner();
                self.custom_plot_handle = Some(handle);
            }
        }
        if let Some(err) = &self.custom_plot_last_err {
            ui.label(err);
        }

        if let Some(df) = &self.custom_plot_data {
            let df = df.drop_nulls::<String>(None).unwrap();
            let columns = df.get_column_names();
            if columns.len() == 2 {
                let c1 = columns[0];
                let c2 = columns[1];
                let x = df.column(c1).unwrap().cast(&DataType::Float64).unwrap();
                let x = x.f64().unwrap().into_no_null_iter();
                let y = df.column(c2).unwrap().cast(&DataType::Float64).unwrap();
                let y = y.f64().unwrap().into_no_null_iter();
                let xy = x.zip(y).map(|(x, y)| [x as f64, y as f64]).collect_vec();
                Plot::new("custom").show(ui, |plot_ui| {
                    plot_ui.points(
                        Points::new(PlotPoints::from(xy))
                            .shape(egui_plot::MarkerShape::Circle)
                            .radius(2.0),
                    );
                });
            }
            if columns.len() == 3 {
                let c1 = columns[0];
                let c2 = columns[1];
                let c3 = columns[2];
                let x = df.column(c1).unwrap().cast(&DataType::Float64).unwrap();
                let x = x.f64().unwrap().into_no_null_iter();
                let y = df.column(c2).unwrap().cast(&DataType::Float64).unwrap();
                let y = y.f64().unwrap().into_no_null_iter();
                let group = df.column(c3).expect("getting group column").rechunk();
                let group = group.as_materialized_series().iter().map(|x| x.to_string());
                Plot::new("custom")
                    .legend(Legend::default())
                    .x_axis_label(c1.to_string())
                    .y_axis_label(c2.to_string())
                    .show(ui, |plot_ui| {
                        x.zip(y)
                            .zip(group)
                            .sorted_by_key(|(_, name)| name.clone())
                            .group_by(|(_, group)| group.to_string())
                            .into_iter()
                            .for_each(|(name, group)| {
                                let xy = group
                                    .map(|((x, y), group)| [x as f64, y as f64])
                                    .collect_vec();

                                plot_ui.points(
                                    Points::new(PlotPoints::from(xy))
                                        .name(name)
                                        .shape(egui_plot::MarkerShape::Circle)
                                        .radius(2.0),
                                );
                            })
                    });
            }
        }
    }

    fn render_plots2(
        &mut self,
        ui: &mut egui::Ui,
        run_ensemble_color: &HashMap<String, egui::Color32>,
    ) {
        let metric_names = self
            .runs2
            .plot_map
            .keys()
            .map(|key| (key.variable.clone(), key.xaxis.clone()))
            .unique()
            .filter(|(name, _)| {
                self.gui_params.filters.metric_filters.contains(name)
                    || self.gui_params.filters.metric_filters.is_empty()
            });

        // let plot_width = ui.available_width() / 2.1;
        // let plot_height = ui.available_width() / 4.1;
        let plot_width = ui.available_width() / 1.1;
        let plot_height = ui.available_width() / 2.1;

        let plots = metric_names
            .map(|name| {
                (
                    name.clone(),
                    Plot::new(name.0)
                        .legend(Legend::default())
                        .width(plot_width)
                        .height(plot_height),
                )
            })
            .collect_vec();

        ui.horizontal_wrapped(|ui| {
            for (metric_name_and_axis, plot) in plots.into_iter().sorted_by_key(|(k, _v)| k.clone())
            {
                ui.allocate_ui(egui::Vec2::from([plot_width, plot_height]), |ui| {
                    ui.vertical_centered(|ui| {
                        ui.label(metric_name_and_axis.0.clone());
                        let plot_res = plot.show(ui, |plot_ui| {
                            for train_id in self.runs2.active_runs.iter().sorted() {
                                if let Some(xys) = self.runs2.plot_map.get(&PlotMapKey {
                                    train_id: train_id.clone(),
                                    variable: metric_name_and_axis.0.clone(),
                                    xaxis: metric_name_and_axis.1.clone(),
                                }) {
                                    for xy in xys {
                                        let xy = xy
                                            .clone()
                                            .iter()
                                            // .map(|[x, y]| [*x, y.max(f64::MIN).log10()])
                                            .map(|[x, y]| [*x as f64, *y as f64])
                                            .collect::<Vec<_>>();
                                        let stroke_width = if let Some(selected_runs) =
                                            &self.gui_params.selected_runs
                                        {
                                            if selected_runs.contains(train_id) {
                                                2.0
                                            } else {
                                                1.0
                                            }
                                        } else {
                                            1.0
                                        };
                                        plot_ui.line(
                                            Line::new(PlotPoints::from(xy.clone()))
                                                .stroke(Stroke::new(
                                                    stroke_width,
                                                    *run_ensemble_color
                                                        .get(train_id)
                                                        .unwrap_or(&egui::Color32::BLACK),
                                                ))
                                                .name(label_from_active_inspect_params(
                                                    train_id.clone(),
                                                    &self.runs2.run_params,
                                                    &self.gui_params,
                                                ))
                                                .id(train_id.clone().into()),
                                        );
                                        plot_ui.points(
                                            Points::new(PlotPoints::from(xy))
                                                .shape(egui_plot::MarkerShape::Circle)
                                                .color(
                                                    *run_ensemble_color
                                                        .get(train_id)
                                                        .unwrap_or(&egui::Color32::BLACK),
                                                ),
                                        );
                                    }
                                }
                            }
                        });
                        if let Some(hovered_id) = plot_res.hovered_plot_item {
                            for train_id in &self.runs2.active_runs.clone() {
                                if egui::Id::from(train_id.clone()) == hovered_id {
                                    // self.runs2.active_runs = vec![train_id.clone()];
                                    self.gui_params.hovered_run = Some(train_id.clone());
                                    if plot_res.response.clicked() {
                                        // self.gui_params.selected_run = Some(train_id.clone());
                                        let set =
                                            self.gui_params.selected_runs.get_or_insert([].into());
                                        if set.contains(train_id) {
                                            set.remove(train_id);
                                        } else {
                                            set.insert(train_id.clone());
                                        }
                                        // .insert(train_id.clone());
                                    }
                                }
                            }
                        }
                        if plot_res.response.hovered() && plot_res.hovered_plot_item.is_none() {
                            self.gui_params.hovered_run = None;
                        }

                        // if plot_res.response.clicked() {
                        //     if let Some(hovered_id) = plot_res.hovered_plot_item {
                        //         for train_id in &self.runs2.active_runs.clone() {
                        //             if egui::Id::from(train_id.clone()) == hovered_id {
                        //                 self.runs2.active_runs = vec![train_id.clone()];
                        //             }
                        //             // self.runs2.active_runs = vec![hovered_id]
                        //         }
                        //     }
                        // }
                    });
                });
            }
        });
        // }
        //     self.runs2.metrics = Some(metrics);
        // }
    }

    // fn render_plots(
    //     &mut self,
    //     ui: &mut egui::Ui,
    //     metric_names: Vec<String>,
    //     run_ensemble_color: &HashMap<String, egui::Color32>,
    // ) {
    //     let xaxis_ids: HashMap<_, _> = self
    //         .runs
    //         .runs
    //         .values()
    //         .map(|run| run.metrics.values().map(|metric| metric.xaxis.clone()))
    //         .flatten()
    //         .unique()
    //         .sorted()
    //         .map(|xaxis| (xaxis.clone(), ui.id().with(xaxis)))
    //         .collect();

    //     let metric_name_axis_id: HashMap<_, _> = self
    //         .runs
    //         .runs
    //         .values()
    //         .map(|run| {
    //             run.metrics.iter().map(|(metric_name, metric)| {
    //                 (metric_name, xaxis_ids.get(&metric.xaxis).unwrap())
    //             })
    //         })
    //         .flatten()
    //         .unique()
    //         .collect();

    //     // let link_group_id = ui.id().with("linked_demo");
    //     let filtered_metric_names: Vec<String> = metric_names
    //         .into_iter()
    //         .filter(|name| {
    //             self.gui_params.filters.metric_filters.contains(name)
    //                 || self.gui_params.filters.metric_filters.is_empty()
    //         })
    //         .collect();
    //     let plot_width = if filtered_metric_names.len() <= 2 {
    //         ui.available_width() / 2.1
    //     } else {
    //         ui.available_width() / 2.1
    //     };
    //     let plot_height = if filtered_metric_names.len() <= 2 {
    //         ui.available_width() / 4.1
    //     } else {
    //         ui.available_width() / 4.1
    //     };
    //     let x_axis = self.gui_params.x_axis.clone();
    //     let formatter = match x_axis {
    //         XAxis::Time => |_name: &str, value: &PlotPoint| {
    //             let ts = NaiveDateTime::from_timestamp_opt(value.x as i64, 0).unwrap();
    //             let xstr = ts.format("%y/%m/%d - %H:%M").to_string();
    //             format!("time: {}\ny: {}", xstr, value.y)
    //         },
    //         XAxis::Batch => {
    //             |_name: &str, value: &PlotPoint| format!("x:{:.3}\ny:{:.3}", value.x, value.y)
    //         }
    //     };
    //     let plots: HashMap<_, _> = filtered_metric_names
    //         .into_iter()
    //         .map(|metric_name| {
    //             (
    //                 metric_name.clone(),
    //                 Plot::new(&metric_name)
    //                     // .auto_bounds()
    //                     .legend(Legend::default())
    //                     .width(plot_width)
    //                     .height(plot_height)
    //                     // .label_formatter(match x_axis {
    //                     //     XAxis::Time => |name, value: &PlotPoint| {
    //                     //         let ts =
    //                     //             NaiveDateTime::from_timestamp_opt(value.x as i64, 0).unwrap();
    //                     //         let xstr = ts.format("%y/%m/%d-%Hh-%Mm").to_string();
    //                     //         format!("time: {}\ny: {}", xstr, value.y)
    //                     //     },
    //                     //     XAxis::Batch => |name, value: &PlotPoint| {
    //                     //         format!("x:{:.3}\ny:{:.3}", value.x, value.y)
    //                     //     },
    //                     // })
    //                     .label_formatter(formatter)
    //                     .x_axis_formatter(match x_axis {
    //                         XAxis::Time => {
    //                             |grid_mark: egui_plot::GridMark,
    //                              // _n_chars,
    //                              range: &RangeInclusive<f64>| {
    //                                 let ts = NaiveDateTime::from_timestamp_opt(
    //                                     grid_mark.value as i64,
    //                                     0,
    //                                 )
    //                                 .unwrap();
    //                                 let delta = range.end() - range.start();
    //                                 if delta > (5 * 24 * 60 * 60) as f64 {
    //                                     ts.format("%m/%d").to_string()
    //                                 } else if delta > (5 * 60 * 60) as f64 {
    //                                     ts.format("%d-%Hh").to_string()
    //                                 } else {
    //                                     ts.format("%Hh:%Mm").to_string()
    //                                 }
    //                             }
    //                         }
    //                         XAxis::Batch => {
    //                             |grid_mark: egui_plot::GridMark,
    //                              // _n_chars,
    //                              _range: &RangeInclusive<f64>| {
    //                                 format!("{}", grid_mark.value as i64).to_string()
    //                             }
    //                         }
    //                     })
    //                     .link_axis(
    //                         **metric_name_axis_id.get(&metric_name).unwrap(),
    //                         true,
    //                         false,
    //                     )
    //                     .link_cursor(**metric_name_axis_id.get(&metric_name).unwrap(), true, true),
    //             )
    //         })
    //         .collect();
    //     // egui::ScrollArea::vertical().show(ui, |ui| {
    //     ui.horizontal_wrapped(|ui| {
    //         for (metric_name, plot) in plots.into_iter().sorted_by_key(|(k, _v)| k.clone()) {
    //             ui.allocate_ui(egui::Vec2::from([plot_width, plot_height]), |ui| {
    //                 ui.vertical_centered(|ui| {
    //                     ui.label(&metric_name);
    //                     plot.show(ui, |plot_ui| {
    //                         if self.gui_params.n_average > 1 {
    //                             for (run_id, run) in
    //                                 self.runs.active_runs.iter().sorted().map(|train_id| {
    //                                     (train_id, self.runs.runs.get(train_id).unwrap())
    //                                 })
    //                             {
    //                                 if let Some(metric) = run.metrics.get(&metric_name) {
    //                                     // let label = self.label_from_active_inspect_params(run);
    //                                     if metric.values.len() == 1 {
    //                                         plot_ui.points(
    //                                             egui_plot::Points::new(PlotPoints::from(
    //                                                 metric.values.clone(),
    //                                             ))
    //                                             .shape(egui_plot::MarkerShape::Circle)
    //                                             .radius(5.0),
    //                                         );
    //                                     }
    //                                     plot_ui.line(
    //                                         Line::new(PlotPoints::from(metric.values.clone()))
    //                                             // .name(&label)
    //                                             .stroke(Stroke::new(
    //                                                 1.0,
    //                                                 run_ensemble_color
    //                                                     .get(run_id)
    //                                                     .unwrap()
    //                                                     .gamma_multiply(0.4),
    //                                             )),
    //                                     );
    //                                 }
    //                             }
    //                         }
    //                         for (run_id, run) in
    //                             self.runs.active_runs.iter().sorted().map(|train_id| {
    //                                 (train_id, self.runs.runs.get(train_id).unwrap())
    //                             })
    //                         {
    //                             if let Some(metric) = run.metrics.get(&metric_name) {
    //                                 let label = "test".into();
    //                                 //label_from_active_inspect_params(run, &self.gui_params);
    //                                 let running = metric.last_value
    //                                     > chrono::Utc::now()
    //                                         .naive_utc()
    //                                         .checked_sub_signed(chrono::Duration::minutes(5))
    //                                         .unwrap();
    //                                 let label = if running {
    //                                     format!("{} (r)", label)
    //                                 } else {
    //                                     label
    //                                 };
    //                                 plot_ui.line(
    //                                     Line::new(PlotPoints::from(metric.resampled.clone()))
    //                                         .name(&label)
    //                                         .stroke(Stroke::new(
    //                                             2.0 * if running { 2.0 } else { 1.0 },
    //                                             *run_ensemble_color.get(run_id).unwrap(),
    //                                         )),
    //                                 );
    //                             }
    //                         }
    //                     })
    //                 });
    //                 // });
    //             });
    //         }
    //     });
    //     // });
    //     if self.data_status == DataStatus::FirstDataProcessed {
    //         // plot_ui.set_auto_bounds(egui::Vec2b::new(true, true));
    //         self.data_status = DataStatus::FirstDataPlotted;
    //     }
    // }

    fn render_metrics(&mut self, ui: &mut egui::Ui, metric_names: &Vec<String>) {
        ui.collapsing("metrics", |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.vertical(|ui| {
                    for metric_name in metric_names {
                        let active_filter =
                            self.gui_params.filters.metric_filters.contains(metric_name);
                        if ui
                            .add(egui::Button::new(metric_name).selected(active_filter))
                            .clicked()
                        {
                            if self.gui_params.filters.metric_filters.contains(metric_name) {
                                self.gui_params.filters.metric_filters.remove(metric_name);
                            } else {
                                self.gui_params
                                    .filters
                                    .metric_filters
                                    .insert(metric_name.clone());
                            }
                            // self.dirty = true;
                            self.update_filtered_runs();
                            self.db_train_runs_sender_slot = Some(self.runs2.active_runs.clone());
                            self.gui_params_sender_slot = Some(self.gui_params.clone());
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
                        .runs2
                        .active_runs
                        // .active_runs
                        .iter()
                        // .map(|train_id| self.runs.runs.get(train_id).unwrap().artifacts.keys())
                        .map(|train_id| {
                            self.runs2
                                .artifacts
                                .iter()
                                .filter(|(k, v)| k.train_id == *train_id)
                                .map(|(k, v)| k.name.clone())
                        })
                        .flatten()
                        .unique()
                        .sorted()
                    // .sorted_by(|a, b| Ord::cmp(&a.name, &b.name))
                    {
                        if ui
                            .add(
                                egui::Button::new(artifact_name.clone()).selected(
                                    self.gui_params
                                        .filters
                                        .artifact_filters
                                        .contains(&artifact_name),
                                ),
                            )
                            .clicked()
                        {
                            if self
                                .gui_params
                                .filters
                                .artifact_filters
                                .contains(&artifact_name)
                            {
                                self.gui_params
                                    .filters
                                    .artifact_filters
                                    .remove(&artifact_name);
                            } else {
                                self.gui_params
                                    .filters
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
        // run_params: &HashMap<(String, String), String>,
    ) {
        let active_artifact_types: Vec<ArtifactType> = self
            .gui_params
            .filters
            .artifact_filters
            .iter()
            .flat_map(|artifact_name| {
                self.runs2.active_runs.iter().map(|train_id| {
                    if let Some(artifact_id) = self.runs2.artifacts.get(&ArtifactKey {
                        train_id: train_id.clone(),
                        name: artifact_name.clone(),
                    }) {
                        let artifact_type = get_artifact_type(&artifact_id.name);
                        if self.artifact_handlers.contains_key(
                            self.artifact_dispatch
                                .get(&artifact_type)
                                .unwrap_or(&ArtifactHandlerType::Unknown),
                        ) {
                            // add_artifact(handler, ui, train_id, path);
                            Some(artifact_type)
                        } else {
                            println!("No handler for {:?} {}", artifact_type, artifact_id.name);
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

        let t = Instant::now();
        for artifact_type in active_artifact_types {
            if let Some(handler) = self.artifact_handlers.get_mut(
                self.artifact_dispatch
                    .get(&artifact_type)
                    .unwrap_or(&ArtifactHandlerType::Unknown),
            ) {
                for train_id in &self.runs2.active_runs {
                    for artifact_id in self
                        .runs2
                        .artifacts_by_run
                        .get(train_id)
                        .unwrap_or(&Vec::new())
                        .iter()
                        .filter(|art_id| {
                            self.gui_params
                                .filters
                                .artifact_filters
                                .contains(&art_id.name)
                        })
                    // run.artifacts.iter().filter(|&(art_name, _)| {
                    //     self.gui_params.filters.artifact_filters.contains(art_name)
                    // })
                    {
                        if artifact_type == get_artifact_type(&artifact_id.name) {
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
                let t = Instant::now();
                show_artifacts(
                    ui,
                    handler,
                    &mut self.gui_params,
                    // &self.runs.runs,
                    &self.runs2.active_runs,
                    &run_ensemble_color,
                    &self.runs2.run_params,
                );
            }
            // if let Some(handler) = self.artifact_handlers.get(artifact_type) {}
        }
    }

    // fn render_time_selector(&mut self, ui: &mut egui::Ui, param_filter: &mut RunsFilter) {
    //     if self.runs.runs_time_ordered.len() > 1 {
    //         ui.vertical(|ui| {
    //             // ui.spacing_mut().slider_width = ui.available_width(); // - 300.0;
    //             ui.label("Cut-off time");
    //             let time_slider = egui::Slider::new(
    //                 &mut param_filter.time_filter_idx,
    //                 0..=self.runs.runs.len() - 1,
    //             )
    //             .custom_formatter(|fval, _| {
    //                 let idx = fval as usize;
    //                 let created_at = self.runs.runs_time_ordered[idx].1;
    //                 created_at.to_string()
    //             });
    //             if ui.add(time_slider).changed() {
    //                 // self.dirty = true;
    //                 param_filter.time_filter =
    //                     Some(self.runs.runs_time_ordered[param_filter.time_filter_idx].1);
    //                 self.update_filtered_runs();
    //                 self.db_train_runs_sender_slot = Some(self.runs.active_runs.clone());
    //                 self.gui_params_sender_slot =
    //                     Some((self.gui_params.clone(), self.runs.runs.clone()));
    //             }
    //         });
    //     }
    // }

    #[instrument(skip(self))]
    fn handle_filtered_runs(&mut self) {
        if let Some(handle) = self.active_runs_handle.take() {
            if handle.is_finished() {
                self.runs2.active_runs =
                    tokio::runtime::Handle::current().block_on(handle).unwrap();
                self.db_train_runs_sender_slot = Some(self.runs2.active_runs.clone());
                println!("updated active runs to {:?}", self.runs2.active_runs);
            } else {
                self.active_runs_handle = Some(handle);
            }
        }
        // for (param_key, v) in self
        //     .runs2
        //     .run_params
        //     .iter()
        //     .filter(|(param_key, v)| self.runs2.active_runs.contains(&param_key.train_id))
        // {
        //     if !self
        //         .gui_params
        //         .filtered_values
        //         .contains_key(&param_key.name)
        //     {
        //         self.gui_params
        //             .filtered_values
        //             .insert(param_key.name.clone(), HashSet::new());
        //     }
        //     self.gui_params
        //         .filtered_values
        //         .get_mut(&param_key.name)
        //         .unwrap()
        //         .insert(v.clone());
        // }
        if self.runs2.active_runs.len() == 0 {
            return;
        }
        {
            // if let Ok(conn) = self.duckdb.get() {
            //     // let stmt = conn.prepare()
            //     let s = tracing::span!(Level::TRACE, "handle_filtered_runs duck");
            //     let _enter = s.enter();
            //     let query = format!(
            //         "
            //         SELECT variable, list(DISTINCT value_text) as values FROM local.runs
            //         WHERE train_id IN ({}) GROUP BY variable
            //         ",
            //         repeat_vars(self.runs2.active_runs.len()),
            //     );
            //     let mut stmt = conn.prepare(&query).unwrap();
            //     let polars = stmt
            //         .query_polars(duckdb::params_from_iter(self.runs2.active_runs.iter()))
            //         .expect("duck artifacts");
            //     drop(_enter);
            //     let polars = polars.collect_vec();
            //     if polars.len() == 0 {
            //         return;
            //     }
            //     let large_df = polars
            //         .into_iter()
            //         .reduce(|acc, e| acc.vstack(&e).unwrap())
            //         .unwrap();
            //     let vars = large_df
            //         .column("variable")
            //         .unwrap()
            //         .as_materialized_series()
            //         .rechunk();
            //     let vars = vars.iter();
            //     let values = large_df
            //         .column("values")
            //         .unwrap()
            //         .as_materialized_series()
            //         .rechunk();
            //     let values = values.iter();
            //     for (var, value) in vars.zip(values) {
            //         if let AnyValue::List(l) = value {
            //             self.gui_params
            //                 .filtered_values
            //                 .entry(var.to_string())
            //                 .insert_entry(l.iter().map(|x| x.to_string()).collect());
            //         }
            //     }
            // }
            // if self.runs2.
            if let Some(handle) = self.new_filtered_values_handle.take() {
                if handle.is_finished() {
                    let s = tracing::span!(Level::TRACE, "handle_filtered_runs finished");
                    let _enter = s.enter();
                    self.gui_params.filtered_values =
                        tokio::runtime::Handle::current().block_on(handle).unwrap();
                } else {
                    self.new_filtered_values_handle = Some(handle);
                }
            }
            if self.new_filtered_values_handle.is_none() {
                let s = tracing::span!(Level::TRACE, "handle_filtered_runs spawn");
                let _enter = s.enter();
                let params = self.runs2.run_params.clone();
                let active_runs = self.runs2.active_runs.clone();
                self.new_filtered_values_handle = Some(tokio::spawn(async move {
                    let mut filtered: HashMap<String, HashSet<String>> = HashMap::new();
                    params
                        .iter()
                        .filter(|(pk, _)| active_runs.contains(&pk.train_id))
                        .for_each(|(pk, v)| {
                            filtered
                                .entry(pk.name.clone())
                                .or_insert_with(HashSet::new)
                                .extend(v.iter().cloned())
                            // .insert(v.clone());
                        });
                    filtered
                }))
            }
        }
    }
    fn update_filtered_runs(&mut self) {
        let pool = self.duckdb.clone();
        let gui_params = self.gui_params.clone();
        self.handle_filtered_runs();
        if self.active_runs_handle.is_some() {
            self.active_runs_handle.take().unwrap().abort();
        }
        self.active_runs_handle = Some(tokio::spawn(async move {
            get_train_ids_from_filter_duck(&pool, &gui_params)
        }));
    }

    #[instrument(skip(self))]
    fn handle_param_values(&mut self) {
        if let Some(handle) = self.param_values_handle.take() {
            if handle.is_finished() {
                self.gui_params.param_values =
                    tokio::runtime::Handle::current().block_on(handle).unwrap();
                println!("handle finished");
            } else {
                self.param_values_handle = Some(handle);
            }
        }
    }
}

fn render_param_tree(
    inspect_filters: &mut HashSet<String>,
    param_filter: &mut RunsFilter,
    filtered_values: &HashMap<String, HashSet<String>>,
    run_params: &HashMap<RunParamKey, HashSet<String>>,
    // runs: &DataFrame,
    tree: Vec<Tree>,
    ui: &mut egui::Ui,
    run_ensemble_color: &HashMap<String, egui::Color32>,
    diff_counts: &HashMap<&str, usize>,
    filter_name_filter: &String,
) {
    let groups = tree.into_iter().group_by(|el| match el {
        Tree::Node { cat, full_cat, .. } => Group::Category(cat.clone(), full_cat.clone()),
        Tree::Leaf(name, _) => Group::Leaf,
    });
    for (key, group) in groups.into_iter() {
        let group = group.collect_vec();
        match key {
            Group::Leaf => {
                for node in group {
                    if let Tree::Leaf(name, path) = node {
                        if !path.contains(filter_name_filter) {
                            continue;
                        }
                        let opened = !param_filter.filter.get(&path).unwrap().is_empty();
                        let force_open = if !filter_name_filter.is_empty() {
                            Some(path.contains(filter_name_filter))
                        } else {
                            None
                        };
                        let label =
                            format!("{}({})", name, diff_counts.get(path.as_str()).unwrap_or(&0));
                        let header_text = if inspect_filters.contains(&path) {
                            egui::RichText::new(label)
                                .underline()
                                .color(egui::Color32::GREEN)
                        } else {
                            egui::RichText::new(label)
                        };
                        fn circle_icon(
                            ui: &mut egui::Ui,
                            openness: f32,
                            response: &egui::Response,
                        ) {
                            let stroke = ui.style().interact(&response).fg_stroke;
                            let radius = egui::lerp(2.0..=3.0, openness);
                            ui.painter().circle_filled(
                                response.rect.center(),
                                radius,
                                stroke.color,
                            );
                        }
                        let header_response = egui::CollapsingHeader::new(header_text)
                            .icon(circle_icon)
                            .default_open(opened)
                            .open(force_open)
                            .show(ui, |ui| {
                                let multival = run_params
                                    .iter()
                                    .filter(|(k, v)| k.name == path)
                                    .flat_map(|(k, v)| v.iter().map(|v| (v, k.train_id.clone())))
                                    .sorted_by_key(|x| x.0)
                                    .group_by(|x| x.0)
                                    .into_iter()
                                    // .unique()
                                    .map(|(v, group)| {
                                        if let Ok(val_int) = v.parse::<i32>() {
                                            (
                                                val_int,
                                                v,
                                                group.map(|x| x.1).sorted().collect::<Vec<_>>(),
                                            )
                                        } else {
                                            (0, v, group.map(|x| x.1).sorted().collect::<Vec<_>>())
                                        }
                                    })
                                    .sorted();
                                ui.horizontal_wrapped(|ui| {
                                    if ui
                                        .selectable_label(inspect_filters.contains(&path), "label")
                                        .clicked()
                                    {
                                        if !inspect_filters.contains(&path) {
                                            inspect_filters.insert(path.clone());
                                        } else {
                                            inspect_filters.remove(&path);
                                        }
                                    }
                                    for (_val_int, val_str, train_ids) in multival.into_iter() {
                                        let has_val = param_filter
                                            .filter
                                            .get(&path)
                                            .unwrap()
                                            .contains(val_str);
                                        let in_filter = filtered_values
                                            .get(&path)
                                            .unwrap_or(&HashSet::new())
                                            .contains(val_str);
                                        // ui.label(val_str);
                                        let border_color = if has_val {
                                            egui::Color32::LIGHT_GREEN
                                        } else {
                                            ui.ctx().style().visuals.widgets.inactive.bg_fill
                                        };
                                        // let bg_color = if has_val {
                                        //     // egui::Color32::LIGHT_BLUE
                                        //     ui.style().visuals.code_bg_color
                                        // } else {
                                        //     ui.ctx().style().visuals.widgets.inactive.bg_fill
                                        // };
                                        let button = ui.add(
                                            egui::Button::new(val_str)
                                                .stroke(egui::Stroke::new(1.0, border_color)), // .fill(bg_color),
                                        );
                                        let (_, color_rect) =
                                            button.rect.split_top_bottom_at_fraction(0.9);
                                        let colors = train_ids
                                            .into_iter()
                                            .filter_map(|train_id| {
                                                run_ensemble_color.get(&train_id)
                                            })
                                            .collect_vec();
                                        if colors.len() > 0 {
                                            let start_x = color_rect.left();
                                            let dx = (color_rect.right() - color_rect.left())
                                                / colors.len() as f32;
                                            for (idx, color) in colors.into_iter().enumerate() {
                                                let x_range = (start_x + dx * idx as f32)
                                                    ..=(start_x + dx * (idx as f32 + 1.0));
                                                let y_range =
                                                    color_rect.top()..=color_rect.bottom();
                                                ui.painter().rect_filled(
                                                    egui::Rect::from_x_y_ranges(x_range, y_range),
                                                    0.0,
                                                    *color,
                                                );
                                            }
                                        }

                                        if button.clicked() {
                                            if has_val {
                                                param_filter
                                                    .filter
                                                    .get_mut(&path)
                                                    .unwrap()
                                                    .remove(val_str);
                                            } else {
                                                param_filter
                                                    .filter
                                                    .get_mut(&path)
                                                    .unwrap()
                                                    .insert(val_str.clone());
                                                dbg!(&path, val_str);
                                            }
                                        }
                                        if button.hovered() {}
                                    }
                                });
                            });
                        if header_response.header_response.secondary_clicked() {
                            if !inspect_filters.contains(&path) {
                                inspect_filters.insert(path.clone());
                            } else {
                                inspect_filters.remove(&path);
                            }
                        }
                    }
                }
            }
            Group::Category(cat, full_cat) => {
                let opened = param_filter
                    .filter
                    .iter()
                    .any(|(key, vals)| key.starts_with(&full_cat) && !vals.is_empty());
                let force_open = if !filter_name_filter.is_empty() {
                    Some(contains_str(&group, &filter_name_filter))
                } else {
                    None
                };
                let has_val = param_filter
                    .filter
                    .iter()
                    .any(|(param, values)| param.starts_with(&full_cat) && !values.is_empty());
                let cat_color = if inspect_filters
                    .iter()
                    .any(|path| path.starts_with(&full_cat))
                {
                    egui::Color32::GREEN
                } else {
                    ui.style().visuals.text_color()
                };
                let mut cat_text = egui::RichText::new(format!(
                    "{} ({})",
                    cat,
                    diff_counts.get(full_cat.as_str()).unwrap_or(&0usize)
                ))
                .color(cat_color);
                if has_val {
                    cat_text = cat_text.underline();
                }
                egui::CollapsingHeader::new(cat_text)
                    .default_open(opened)
                    .open(force_open)
                    .show(ui, |ui| {
                        let subtree = group
                            .iter()
                            .map(|el| {
                                if let Tree::Node {
                                    rest,
                                    path,
                                    full_cat,
                                    ..
                                } = el
                                {
                                    match rest.split_once(".") {
                                        Some((cat, rest)) => Tree::Node {
                                            cat: cat.to_string(),
                                            rest: rest.to_string(),
                                            path: path.clone(),
                                            full_cat: format!("{full_cat}.{cat}"),
                                        },
                                        None => Tree::Leaf(rest.clone(), path.clone()),
                                    }
                                } else {
                                    Tree::Leaf("error".to_string(), "error".to_string())
                                }
                            })
                            .collect_vec();
                        render_param_tree(
                            inspect_filters,
                            param_filter,
                            filtered_values,
                            run_params,
                            // runs,
                            subtree,
                            ui,
                            run_ensemble_color,
                            diff_counts,
                            filter_name_filter,
                        );
                    });
                // ui.label(cat);
            }
        }
    }
}

#[instrument(skip_all)]
fn get_parameter_values_duck(
    pool: &r2d2::Pool<DuckdbConnectionManager>,
) -> HashMap<String, HashSet<String>> {
    let mut conn = pool.get().unwrap();

    // conn.is_autocommit()
    let tx = conn.transaction().unwrap();
    //"SELECT DISTINCT name as variable, value FROM local.{table_name} ORDER BY name, value"
    let sql = format!(
        "
        SELECT DISTINCT variable, value_text FROM (
        SELECT *, name as variable, value as value_text
        FROM local.model_parameter_text JOIN local.models ON id=model_id
        -- WHERE name NOT IN () 
        UNION
        SELECT *, name as variable, format('{{:E}}', value) as value_text
        FROM local.model_parameter_float JOIN local.models ON id=model_id
         -- WHERE name NOT IN () 
        UNION
        SELECT *, name as variable, format('{{:d}}', value) as value_text
        FROM local.model_parameter_int JOIN local.models ON id=model_id
        --WHERE name NOT IN ()
        )
        ORDER BY variable, value_text"
    );
    let mut stmt = tx.prepare(sql.as_str()).unwrap();
    let rows = stmt
        .query_map([], |row| {
            duckdb::Result::Ok((
                row.get::<_, String>(0).unwrap(),
                row.get::<_, String>(1).unwrap(),
            ))
        })
        .unwrap()
        .map(|row| row.unwrap())
        .group_by(|(a, b)| a.clone());
    let rows = rows
        .into_iter()
        .map(|(key, group)| (key, group.map(|(k, v)| v).collect()));
    //     .map(|row| {
    //         let (a, b) = row.unwrap();
    //         (
    //             a.unwrap(),
    //             b.unwrap()
    //                 .split(",")
    //                 .map(|x| x.to_string())
    //                 .collect::<HashSet<T>>(),
    //         )
    //     });
    // .collect();
    let ret = HashMap::from_iter(rows);
    ret
}

#[instrument(skip_all)]
fn get_parameter_values_duck_old<T: FromSql + std::cmp::Eq + Hash>(
    pool: &r2d2::Pool<DuckdbConnectionManager>,
    table_name: &str,
) -> HashMap<String, HashSet<T>> {
    let conn = pool.get().unwrap();
    let sql = format!(
        "SELECT DISTINCT name as variable, value FROM local.{table_name} ORDER BY name, value"
    );
    let mut stmt = conn.prepare(sql.as_str()).unwrap();
    let rows = stmt
        .query_map([], |row| {
            duckdb::Result::Ok((
                row.get::<_, String>(0).unwrap(),
                row.get::<_, T>(1).unwrap(),
            ))
        })
        .unwrap()
        .map(|row| row.unwrap())
        .group_by(|(a, b)| a.clone());
    let rows = rows
        .into_iter()
        .map(|(key, group)| (key, group.map(|(k, v)| v).collect()));
    //     .map(|row| {
    //         let (a, b) = row.unwrap();
    //         (
    //             a.unwrap(),
    //             b.unwrap()
    //                 .split(",")
    //                 .map(|x| x.to_string())
    //                 .collect::<HashSet<T>>(),
    //         )
    //     });
    // .collect();
    return HashMap::from_iter(rows);
}

// fn get_parameter_values(
//     runs: &Runs, // active_runs: &Vec<String>,
//     all: bool,
// ) -> HashMap<String, HashSet<String>> {
//     let train_ids: Vec<String> = if all {
//         runs.runs.keys().cloned().collect()
//     } else {
//         runs.active_runs.clone()
//     };
//     let mut param_values: HashMap<String, HashSet<String>> = train_ids
//         .iter()
//         .map(|train_id| runs.runs.get(train_id).unwrap())
//         .map(|run| run.params.keys().cloned())
//         .flatten()
//         .unique()
//         .map(|param_name| (param_name, HashSet::with_capacity(train_ids.len())))
//         .collect();
//     for run in train_ids
//         .iter()
//         .map(|train_id| runs.runs.get(train_id).unwrap())
//     {
//         for (k, v) in &run.params {
//             let values = param_values.get_mut(k).unwrap();
//             values.insert(v.clone());
//         }
//     }
//     param_values
// }

async fn get_last_runs_log(
    pool: &sqlx::Pool<sqlx::Postgres>,
    // runs: &mut HashMap<String, Run>,
    // tx_runs: Option<&SyncSender<HashMap<String, Run>>>,
) -> Result<(), sqlx::Error> {
    // let mut new_epoch_timestamp = last_timestamp.clone();
    let q = format!(
        r#"
SELECT train_id,
       CURRENT_TIMESTAMP - MAX(created_at) AS time_since_last
FROM (
    SELECT train_id, created_at
    FROM metrics
    ORDER BY created_at DESC limit 100
) AS sorted_table
GROUP BY train_id limit 10;
        "#,
    );
    let runs_with_time_rows = sqlx::query(q.as_str())
        // .bind(offset)
        .fetch_all(pool)
        .await?;
    Ok(())
}

fn get_artifacts_duck(
    pool: &r2d2::Pool<DuckdbConnectionManager>,
    active_runs: &Vec<String>,
) -> HashMap<ArtifactKey, ArtifactId> {
    // return HashMap::new();
    if active_runs.is_empty() {
        return HashMap::new();
    }
    let mut conn = pool.get().unwrap();
    let conn = conn.transaction().unwrap();
    let query = format!(
        "
        SELECT * EXCLUDE(id), local.artifacts.id as id FROM local.artifacts
        JOIN local.models as t2 ON local.artifacts.model_id=t2.id
        WHERE train_id IN ({}) ORDER BY train_id
        ",
        repeat_vars(active_runs.len()),
    );
    let mut stmt = conn.prepare(&query).unwrap();
    let polars = stmt
        .query_polars(duckdb::params_from_iter(active_runs))
        .expect("duck artifacts");
    let polars = polars.collect_vec();
    if polars.len() == 0 {
        return HashMap::new();
    }
    let large_df = polars
        .into_iter()
        .reduce(|acc, e| acc.vstack(&e).unwrap())
        .unwrap();
    large_df.sort(vec!["train_id"], Default::default()).unwrap();

    let train_ids = large_df.column("train_id").unwrap().rechunk();
    let train_ids = train_ids.as_materialized_series().iter();
    let names = large_df.column("name").unwrap().rechunk();
    let names = names.as_materialized_series().iter();
    let ids = large_df.column("id").unwrap().rechunk();
    let ids = ids.as_materialized_series().iter();
    let ret = train_ids
        .zip(names)
        .zip(ids)
        .map(|((tid, name), id)| {
            let id = match id {
                AnyValue::Int64(id) => id,
                _ => panic!(),
            };
            (
                ArtifactKey {
                    train_id: tid.get_str().unwrap().to_string(),
                    name: name.get_str().unwrap().to_string(),
                },
                ArtifactId {
                    artifact_id: id,
                    train_id: tid.get_str().unwrap().to_string(),
                    name: name.get_str().unwrap().to_string(),
                    artifact_type: get_artifact_type(&name.get_str().unwrap().to_string()),
                },
            )
        })
        .collect::<HashMap<_, _>>();
    ret
}

fn ensure_duckdb_schema(pool: &r2d2::Pool<DuckdbConnectionManager>) {
    let conn = pool.get().unwrap();
    for table_name in [
        "models",
        "model_parameter_int",
        "model_parameter_float",
        "model_parameter_text",
        "train_step_metric_int",
        "train_step_metric_float",
        "train_step_metric_text",
        "checkpoint_sample_metric_int",
        "checkpoint_sample_metric_float",
        "checkpoint_sample_metric_text",
        "checkpoints",
        "artifacts",
        "artifact_chunks",
        "train_steps",
    ] {
        conn.execute(
            format!(
                "CREATE TABLE IF NOT EXISTS local.{table_name} AS FROM db.{table_name} LIMIT 0"
            )
            .as_str(),
            [],
        )
        .expect(format!("create table {table_name}").as_str());
        // conn.execute("COPY FROM DATABASE db TO local (SCHEMA)", [])
        //     .expect("copy schema");
    }
    conn.execute("SET pg_experimental_filter_pushdown=true", [])
        .expect("set filter push down");
    conn.execute("SET pg_use_ctid_scan=false", [])
        .expect("set ctid scan false");
}

fn repeat_vars(count: usize) -> String {
    assert_ne!(count, 0);
    let mut s = "?,".repeat(count);
    // Remove trailing comma
    s.pop();
    s
}

#[instrument(skip(pool))]
fn get_runs_duck(pool: &r2d2::Pool<DuckdbConnectionManager>) -> polars::prelude::DataFrame {
    let mut conn = pool.get().unwrap();
    let conn = conn.transaction().unwrap();
    let query = format!(
        "
        SELECT * , name as variable, value as value_text
        FROM local.model_parameter_text JOIN local.models ON id=model_id
        WHERE name NOT IN ({}) 
        UNION
        SELECT * , name as variable, format('{{:E}}', value) as value_text
        FROM local.model_parameter_float JOIN local.models ON id=model_id
        WHERE name NOT IN ({}) 
        UNION
        SELECT * , name as variable, format('{{:d}}', value) as value_text
        FROM local.model_parameter_int JOIN local.models ON id=model_id
        WHERE name NOT IN ({}) 
        ",
        repeat_vars(HIDDEN_PARAMS.len()),
        repeat_vars(HIDDEN_PARAMS.len()),
        repeat_vars(HIDDEN_PARAMS.len()),
    );
    let mut stmt = conn.prepare(&query).unwrap();
    let polars = stmt
        .query_polars(duckdb::params_from_iter(
            HIDDEN_PARAMS
                .iter()
                .chain(HIDDEN_PARAMS.iter())
                .chain(HIDDEN_PARAMS.iter()),
        ))
        .expect("duck runs");
    let large_df = polars.reduce(|acc, e| acc.vstack(&e).unwrap()).unwrap();
    large_df
        .sort(vec!["variable", "train_id"], Default::default())
        .unwrap()
}

fn get_run_params(
    pool: &r2d2::Pool<DuckdbConnectionManager>,
) -> HashMap<RunParamKey, HashSet<String>> {
    let t = Instant::now();

    let df = get_runs_duck(pool);
    let t = Instant::now();

    let train_ids = df.column("train_id").unwrap().rechunk();
    let train_ids = train_ids.as_materialized_series().iter();
    let variables = df.column("variable").unwrap().rechunk();
    let variables = variables.as_materialized_series().iter();
    let values = df.column("value_text").unwrap().rechunk();
    let values = values.as_materialized_series().iter();
    let ret = train_ids
        .zip(variables)
        .zip(values)
        .map(|((tid, var), val)| {
            (
                RunParamKey {
                    train_id: tid.get_str().unwrap().to_string(),
                    name: var.get_str().unwrap().to_string(),
                },
                val.get_str().unwrap().to_string(),
            )
        })
        // .into_iter()
        .group_by(|(k, v)| k.clone())
        .into_iter()
        .map(|(name, group)| (name.clone(), group.map(|(_k, v)| v).collect()))
        .collect::<HashMap<_, HashSet<String>>>();
    ret
}

// fn get_run_label_duck(pool: &r2d2::Pool<DuckdbConnectionManager>, gui_params: &GuiParams) {
//     let conn = pool.get().unwrap();
//     let query = format!(
//         "
//         SELECT * FROM local.runs
//         WHERE variable NOT IN ({})
//         ",
//         repeat_vars(HIDDEN_PARAMS.len()),
//     );
//     let mut stmt = conn.prepare(&query).unwrap();

// }

#[instrument(skip(pool))]
fn sync_runs_duck(pool: &r2d2::Pool<DuckdbConnectionManager>) {
    // ensure_duckdb_schema(pool);
    sync_full_table(&pool, "models");
    sync_full_table(&pool, "model_parameter_float");
    sync_full_table(&pool, "model_parameter_int");
    sync_full_table(&pool, "model_parameter_text");
}

#[instrument(skip_all)]
fn sync_full_table(pool: &r2d2::Pool<DuckdbConnectionManager>, table_name: &str) {
    let conn = pool.get().unwrap();
    let max_id: i32 = conn
        .query_row(
            format!("SELECT COALESCE(max(id_serial), 0) FROM local.{table_name}").as_str(),
            [],
            |row| row.get(0),
        )
        .unwrap();
    let query = format!(
        "
        INSERT INTO local.{table_name}
        SELECT * FROM db.{table_name} 
        WHERE id_serial > {}
        ORDER BY id_serial ASC
        ",
        max_id
    );
    conn.execute(&query, [])
        .expect(format!("sync {table_name} failed").as_str());
}

#[instrument(skip_all)]
fn get_metrics_duck(
    pool: &r2d2::Pool<DuckdbConnectionManager>,
    active_runs: &Vec<String>,
) -> polars::prelude::DataFrame {
    if active_runs.len() == 0 {
        return DataFrame::empty();
    }
    let conn = pool.get().unwrap();
    let query = format!(
        // "
        // SELECT train_id, variable, FLOOR(x / 1000.0) as bucket, AVG(x) as x, AVG(value) FROM local.metrics
        // WHERE xaxis='batch' AND train_id IN ({})
        // GROUP BY (train_id, variable, bucket)
        // ORDER BY train_id, variable, x, value
        // ",
        "
            WITH range_info AS (
             SELECT
                 model_id,
                 name,
                 MIN(step) AS min_x,
                 MAX(step) AS max_x
             FROM local.train_step_metric_float
             -- WHERE xaxis='batch'
             GROUP BY model_id, name 
         ),
         bucket_size AS (
             SELECT
                 model_id,
                 name,
                 (max_x - min_x) / 1000.0 AS size
             FROM range_info
         ),
         bucket_table AS (
         SELECT
             -- t.model_id as model_id,
             t.model_id as model_id,
             any_value(m.train_id) as train_id,
             t.name as name,
             FLOOR(t.step / bs.size) AS bucket,
             AVG(t.value) AS value,
             (bucket * ANY_VALUE(bs.size))::DOUBLE as x
             -- AVG(t.value) AS value, AVG(t.x) as x
         FROM local.train_step_metric_float t
         JOIN bucket_size bs
         ON t.model_id = bs.model_id AND t.name=bs.name
         JOIN local.models m
         ON t.model_id = id
         WHERE m.train_id IN ({})
         GROUP BY t.model_id, t.name, bucket
         ORDER BY t.model_id, t.name, x)
         SELECT
             model_id, any_value(train_id) as train_id, name, array_agg(x ORDER BY x) as xs, array_agg(value ORDER BY x) as values
            FROM bucket_table
            GROUP BY model_id, name
        ",
        repeat_vars(active_runs.len()),
    );
    let mut stmt = conn.prepare(&query).unwrap();
    let polars = stmt
        .query_polars(duckdb::params_from_iter(active_runs.iter()))
        .unwrap();
    let large_df = polars.reduce(|acc, e| acc.vstack(&e).unwrap());
    let df = large_df.unwrap_or(DataFrame::empty());
    df
}

#[instrument(skip_all)]
fn sync_table(
    pool: &r2d2::Pool<DuckdbConnectionManager>,
    table_name: &str,
    active_runs: &Vec<String>,
    limit: &mut usize,
) -> bool {
    let conn = pool.get().unwrap();
    let mut updated = false;
    println!("syncing {:?}", active_runs);
    for train_id in active_runs {
        let mut stmt = conn
            .prepare(format!("SELECT id as model_ids from local.models WHERE train_id=?").as_str())
            .unwrap();
        let model_ids: Vec<i64> = stmt
            .query_map([train_id], |row| row.get(0))
            .unwrap()
            .into_iter()
            .map(|x| x.unwrap())
            .collect();
        if model_ids.len() == 0 {
            continue;
        }
        println!("getting max id for {}", train_id);
        let max_id: i32 = conn
            .query_row(
                format!(
                    "SELECT COALESCE(MAX(local.{table_name}.id_serial), 0) from local.{table_name} JOIN local.models ON local.{table_name}.model_id=local.models.id WHERE local.models.train_id=?"
                )
                .as_str(),
                [train_id],
                |row| row.get(0),
            )
            .unwrap();
        println!("[{}] max id {}", train_id, max_id);
        // println!("max id {} for {}", max_id, train_id);
        let query = format!(
            "
        INSERT INTO local.{table_name} BY NAME
        SELECT * 
            EXCLUDE (id, train_id, created_at, id_serial),
            db.{table_name}.id_serial as id_serial,
            db.{table_name}.created_at as created_at 
        FROM db.{table_name}
        JOIN db.models ON id=model_id
        WHERE db.{table_name}.id_serial > {}
        AND
        train_id=?
        LIMIT {}
        ",
            max_id,
            // repeat_vars(model_ids.len()),
            limit
        );
        // println!("{query} {}", train_id);
        let t = Instant::now();
        let n_rows = conn
            .execute(&query, [train_id]) //params_from_iter(model_ids.iter()))
            .expect("sync runs failed");
        println!("Inserted {}", n_rows);
        let max_id_post: i32 = conn
            .query_row(
                format!(
                    "SELECT COALESCE(MAX(local.{table_name}.id_serial), 0) from local.{table_name} JOIN local.models ON local.{table_name}.model_id=models.id WHERE models.train_id=?"
                )
                .as_str(),
                [train_id],
                |row| row.get(0),
            )
            .unwrap();
        if max_id_post != max_id {
            updated = true;
        }
        if max_id_post != max_id && t.elapsed().as_millis() < 500 && *limit < 10e6 as usize {
            *limit = *limit * 2;
            println!("increasing metric limit to {}", limit);
        }
        if max_id_post != max_id && t.elapsed().as_millis() > 2000 {
            *limit = (*limit / 2).max(100000);
            println!("decreasing metric limit to {}", limit);
        }
    }
    updated
}

#[derive(Debug)]
enum ArtifactTransfer {
    Done(Vec<u8>),
    Loading(usize, usize),
    Err(String),
}
// #[tokio::main(flavor = "current_thread")]
fn main() -> Result<(), sqlx::Error> {
    use tracing_subscriber::layer::SubscriberExt;
    tracing::subscriber::set_global_default(
        tracing_subscriber::registry().with(tracing_tracy::TracyLayer::default()),
    )
    .expect("setup tracy layer");
    use tracy_client::Client;
    let _profile_guard = Client::start();
    // console_subscriber::init();
    // Load environment variables
    // dotenv::dotenv().ok();
    let args = Args::parse();
    println!("Args: {:?}", args);
    // let (tx, rx_new_runs_from_db) = mpsc::sync_channel(1);
    let (tx_gui_dirty, rx_gui_dirty) = mpsc::sync_channel(100);
    // let (tx_gui_recomputed, rx_gui_recomputed) = mpsc::sync_channel(100);
    let (tx_new_train_runs_to_db, rx_new_train_runs_to_db) = mpsc::sync_channel::<Vec<String>>(1);
    let (tx_active_runs, rx_active_runs) = mpsc::sync_channel::<Vec<String>>(1);
    let (tx_plot_map, rx_plot_map) = mpsc::sync_channel::<(
        HashMap<ArtifactKey, ArtifactId>,
        HashMap<PlotMapKey, Vec<Vec<[f32; 2]>>>,
    )>(1);
    let (tx_run_params, rx_run_params) =
        mpsc::sync_channel::<HashMap<RunParamKey, HashSet<String>>>(1);
    // let rx_db_filters_am = Arc::new(std::sync::Mutex::new(rx_new_train_runs_to_db));
    let (tx_db_artifact, rx_db_artifact) = mpsc::sync_channel::<ArtifactTransfer>(1);
    let (tx_db_artifact_path, rx_db_artifact_path) = mpsc::sync_channel::<i64>(1);
    let (tx_batch_status, rx_batch_status) = mpsc::sync_channel::<(usize, usize)>(100);
    let rt = tokio::runtime::Runtime::new().unwrap();
    let rt_handle = rt.handle().clone();
    let _guard = rt.enter();

    let manager = DuckdbConnectionManager::memory().unwrap();
    let duckdb_pool = r2d2::Pool::builder().build(manager).unwrap();
    let conn = duckdb_pool.get().unwrap();
    let threads: i32 = conn
        .query_row(
            "
            SELECT value
FROM duckdb_settings()
WHERE name = 'threads';
        ",
            [],
            |row| row.get(0),
        )
        .expect("check threads");
    println!("duckdb threads {}", threads);
    let version: String = conn
        .query_row(
            "
            SELECT version() as version;
        ",
            [],
            |row| row.get(0),
        )
        .expect("check threads");
    println!("duckdb version {}", version);
    conn.execute("INSTALL ducklake;", [])
        .expect("install ducklake failed");
    conn.execute("INSTALL postgres;", [])
        .expect("install postgres failed");
    conn.execute("INSTALL aws;", [])
        .expect("install aws failed");
    conn.execute("CALL load_aws_credentials()", [])
        .expect("load aws credentials");
    // .expect("load postgres failed");
    // conn.execute("ATTACH 'local.db' as local;", [])
    // .expect("attach to psql failed");
    //
    let dburl = env::var("PG").unwrap_or("localhost".into());
    conn.execute(format!("ATTACH 'ducklake:postgres:dbname=ducklake user=postgres password=herdeherde host={dburl} port=5430' as local (data_path 's3://eqp.ducklake')").as_str(), []).expect("attach ducklake");
    conn.execute("USE local", []).expect("attach ducklake");
    conn.execute("SET memory_limit = '1GB';", [])
        .expect("attach to psql failed");
    // if env::var("DATABASE_URL").is_ok() {
    // conn.execute(
    // "ATTACH 'dbname=equiv_v2 user=postgres password=herdeherde host=127.0.0.1 port=5430' as db (TYPE POSTGRES, READ_ONLY);",
    // [],
    // ).expect("attach to psql failed");
    // sync_runs_duck(&duckdb_pool);
    // sync_runs_duck(&duckdb_pool);
    // let db_thread_pool = duckdb_pool.clone();
    // std::thread::spawn(move || {
    //     let mut train_runs = Vec::new();
    //     let mut limit = 1000;
    //     loop {
    //         let start = Instant::now();
    //         // sync_runs_duck(&db_thread_pool);
    //         let start = Instant::now();
    //         if let Ok(new_train_ids) = rx_new_train_runs_to_db.try_recv() {
    //             train_runs = new_train_ids;
    //             // limit = 1000;
    //             println!("[db] got new train runs {:?}", train_runs);
    //         }
    //         let t = Instant::now();
    //         // let updated1 = sync_table(
    //         //     &db_thread_pool,
    //         //     "train_step_metric_float",
    //         //     &train_runs,
    //         //     &mut limit,
    //         // );
    //         // let updated2 = sync_table(&db_thread_pool, "train_steps", &train_runs, &mut limit);
    //         // if updated1 || updated2 {
    //         //     println!("sync elapsed {}", t.elapsed().as_millis());
    //         // }
    //         if start.elapsed().as_millis() < 1000 {
    //             sleep(time::Duration::from_millis(1000) - start.elapsed());
    //         }
    //     }
    // });
    use tracing::Instrument;
    let db_artifact_pool = duckdb_pool.clone();
    let future = async move {
        // let database_url = env::var("DATABASE_URL")
        //     .unwrap_or("postgres://postgres:herdeherde@localhost:5431/equiv".to_string());
        // let pool = PgPoolOptions::new()
        //     .max_connections(5)
        //     .connect(&database_url)
        //     .await
        //     .expect("Can't connect to database");
        println!("Artifact loop starting...");
        loop {
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            if let Ok(artifact_id) = rx_db_artifact_path.try_recv() {
                handle_artifact_request(artifact_id, &db_artifact_pool, &tx_db_artifact).await;
            }
        }
    };
    rt_handle.spawn(future.instrument(tracing::info_span!("handle_artifacts")));
    // }
    let plot_map_thread_pool = duckdb_pool.clone();
    std::thread::spawn(move || loop {
        let start = Instant::now();
        if let Ok(new_active_runs) = rx_active_runs.try_recv() {
            println!("updating...");
            let plot_map = update_plot_map(&plot_map_thread_pool, &new_active_runs);
            let artifacts = get_artifacts_duck(&plot_map_thread_pool, &new_active_runs);
            // let artifacts = if env::var("DATABASE_URL").is_ok() {
            //     ensure_duckdb_schema(&plot_map_thread_pool);
            //     //sync_full_table(&plot_map_thread_pool, "artifacts");
            //     get_artifacts_duck(&plot_map_thread_pool, &new_active_runs)
            // } else {
            //     HashMap::new()
            // };
            tx_plot_map.send((artifacts, plot_map)).unwrap();
        }
        if start.elapsed().as_millis() < 1000 {
            sleep(time::Duration::from_millis(1000) - start.elapsed());
        }
    });

    let run_params_thread_pool = duckdb_pool.clone();
    std::thread::spawn(move || loop {
        let start = Instant::now();
        let params = get_run_params(&run_params_thread_pool);
        tx_run_params.send(params).expect("send run params");
        if start.elapsed().as_millis() < 1000 {
            sleep(time::Duration::from_millis(1000) - start.elapsed());
        }
    });

    let options = eframe::NativeOptions {
        // default_theme: eframe::Theme::Light,
        // initial_window_size: Some(egui::vec2(320.0, 240.0)),
        // initial_window_pos: Some((200., 200.).into()),
        viewport: egui::ViewportBuilder::default().with_inner_size([800.0, 800.0]),
        wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
            wgpu_setup: WgpuSetup::CreateNew(WgpuSetupCreateNew {
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
                        ..Default::default()
                    }
                }),
                ..Default::default()
            }), // supported_backends: wgpu::Backends::VULKAN,
            ..Default::default()
        },
        ..Default::default()
    };
    // let active_runs = vec![
    //     "3cd07788c17d0cd6fdb4d6a10893fd93".to_string(),
    //     "d71d366dfb4902607e8351284ce247e5".to_string(),
    //     "c56b5b515c647089b36ffd4c6e98f343".to_string(),
    // ];
    // let df_metrics = get_state_metrics_duck(&duckdb_pool, &active_runs).unwrap();
    // use polars::lazy::prelude::*;
    // let t = Instant::now();
    // let df = df
    //     .lazy()
    //     .filter(col("variable").eq(lit("loss_batch")))
    //     .filter(col("train_id").eq(lit("3cd07788c17d0cd6fdb4d6a10893fd93")))
    //     .sort("x", Default::default())
    //     .collect()
    //     .expect("filter");
    // let t = Instant::now();
    // let df = df
    //     .lazy()
    //     .filter(col("variable").eq(lit("loss_batch")))
    //     .filter(col("train_id").eq(lit("3cd07788c17d0cd6fdb4d6a10893fd93")))
    //     .sort("x", Default::default())
    //     .collect()
    //     .expect("filter");
    // panic!();
    // let t = df.lazy()
    //     .filter(
    //         &df.column("train_id")
    //             .unwrap()
    //             .equal("3cd07788c17d0cd6fdb4d6a10893fd93")
    //             .unwrap(), // .eq("3cd07788c17d0cd6fdb4d6a10893fd93"),
    //     )
    //     .unwrap();
    // t.column("")
    // t.wind
    // t.group_by_rolling(, )
    // let sampled = t.sample_n_literal(1000, false, false, None).unwrap();
    let runs2 = Runs2 {
        active_runs: Vec::new(),
        plot_map: HashMap::new(),
        run_params: HashMap::new(),
        artifacts: HashMap::new(),
        artifacts_by_run: HashMap::new(),
    };
    let mut filters: Option<Filters> = None;
    if let Ok(mut file) = File::open("last_filters.ron") {
        let mut content = String::new();
        if let Ok(_) = file.read_to_string(&mut content) {
            filters = Some(ron::from_str(&content).expect("Failed to deserialize"));
        }
    }
    let _ = eframe::run_native(
        "Visualizer",
        options,
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            let mut gui_runs = GuiRuns {
                duckdb: duckdb_pool,
                runs2,
                // dirty: true,
                db_train_runs_sender: tx_new_train_runs_to_db,
                rx_plot_map,
                tx_active_runs,
                // recomputed_reciever: rx_gui_recomputed,
                gui_params_sender: tx_gui_dirty,
                initialized: false,
                gui_params: GuiParams {
                    max_n: 1000,
                    n_average: 0,
                    // time_filter: None,
                    // time_filter_idx: 0,
                    x_axis: XAxis::Batch,
                    // param_name_filter: "".to_string(),
                    // table_sorting: HashSet::new(),
                    npy_plot_size: 0.48,
                    render_format: cc.wgpu_render_state.as_ref().unwrap().target_format,
                    param_values: HashMap::new(),
                    filtered_values: HashMap::new(),
                    next_param_update: std::time::Instant::now(),
                    filters: filters.unwrap_or(Filters {
                        param_filters: Vec::new(),
                        metric_filters: HashSet::new(),
                        inspect_params: HashSet::new(),
                        artifact_filters: HashSet::new(),
                    }),
                    hovered_run: None,
                    selected_runs: None,
                },
                // texture: None,
                artifact_handlers: HashMap::from([
                    (
                        ArtifactHandlerType::SpatialNPY,
                        ArtifactHandler::NPYArtifact {
                            arrays: HashMap::new(),
                            textures: HashMap::new(),
                            views: HashMap::new(),
                            _colormap_artifacts: HashSet::new(),
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
                            image_size: 0.9,
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
                    (ArtifactType::NPYSpeced, ArtifactHandlerType::SpecedNPY),
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
                plot_timer: Instant::now() + std::time::Duration::from_secs(2),
                rx_run_params,
                param_values_handle: None,
                active_runs_handle: None,
                filter_save_name: String::new(),
                filter_load_dialog: None,
                table_filter: String::new(),
                custom_plot: String::new(),
                custom_plot_last_err: None,
                custom_plot_data: None,
                custom_plot_handle: None,
                filter_name_filter: String::new(),
                new_filtered_values_handle: None,
            };
            gui_runs.update_filtered_runs();
            gui_runs.db_train_runs_sender_slot = Some(gui_runs.runs2.active_runs.clone());
            gui_runs.gui_params_sender_slot = Some(gui_runs.gui_params.clone());
            // gui_runs.update_plot_map();

            Ok(Box::<GuiRuns>::new(gui_runs))
        }),
    )
    .unwrap();

    Ok(())
}

// #[profiling::function]
#[instrument(skip(pool))]
async fn handle_artifact_request(
    artifact_id: i64,
    // pool: &sqlx::Pool<sqlx::Postgres>,
    pool: &r2d2::Pool<DuckdbConnectionManager>,
    tx_db_artifact: &SyncSender<ArtifactTransfer>,
) {
    println!("[db] artifact requested: {}", artifact_id);
    let mut conn = pool.get().unwrap();
    let conn = conn.transaction().unwrap();
    let filesize = conn
        .query_row(
            "SELECT size FROM local.artifacts WHERE id=?",
            [artifact_id],
            |row| row.get::<_, i64>(0),
        )
        .unwrap();
    println!("Got request for existing artifact id of size {}", filesize);
    let mut last_seq_num: i32 = -1;
    // let mut chunk_size = 1_000_000;
    let mut batch_size: usize = 1;
    let mut buffer: Vec<u8> = vec![0; filesize as usize];
    let mut offset: usize = 0;
    // while offset < filesize {
    loop {
        // let length = chunk_size.min(filesize - offset);
        let query_time = std::time::Instant::now();
        println!("loop step {}, {}", batch_size, last_seq_num);
        // https://github.com/duckdb/duckdb-postgres/issues/233
        let mut stmt = conn.prepare(
                format!("SELECT seq_num, data, size FROM local.artifact_chunks WHERE artifact_id={artifact_id} AND seq_num > {last_seq_num} ORDER BY seq_num LIMIT {batch_size}").as_str()
            ).unwrap();
        // .bind(&artifact_id)
        // .bind(&last_seq_num)
        // .bind(&batch_size)
        // .fetch_all(pool)
        // .await;
        // let res = stmt.query(duckdb::params![artifact_id, last_seq_num, batch_size]);
        let res = stmt.query([]);
        if let Err(err) = &res {
            println!("{:?}", err);
            break;
        }
        if let Ok(mut rows) = res {
            // if rows.
            let mut received_rows = 0;
            while let Some(row) = rows.next().unwrap() {
                let chunk_size: i32 = row.get("size").unwrap();
                let dst = &mut buffer[offset..offset + chunk_size as usize];
                dst.copy_from_slice(row.get::<_, Vec<u8>>("data").unwrap().as_slice());
                offset += chunk_size as usize;
                last_seq_num = row.get::<_, i32>("seq_num").unwrap();
                received_rows += 1;
                println!("[f] Read chunk {} at {}", chunk_size, offset);
                tx_db_artifact
                    .send(ArtifactTransfer::Loading(
                        offset as usize,
                        filesize as usize,
                    ))
                    .unwrap();
            }
            if received_rows == 0 {
                println!("[db] Fetched all available artifact chunks");
                break;
            }
        }
        let elapsed_seconds = query_time.elapsed().as_secs_f32();
        if elapsed_seconds < 0.5 {
            batch_size *= 2;
        } else if elapsed_seconds > 2.0 {
            batch_size = (batch_size / 2).max(1);
        }
    }
    tx_db_artifact.send(ArtifactTransfer::Done(buffer)).unwrap();
}
