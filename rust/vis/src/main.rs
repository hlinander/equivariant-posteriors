use chrono::NaiveDate;
use chrono::NaiveDateTime;
use eframe::egui;
use egui::{epaint, Stroke};
use egui_plot::{Legend, PlotPoints};
use egui_plot::{Line, Plot};
use itertools::Itertools;
use sqlx::postgres::PgPoolOptions;
use sqlx::Row;
// use sqlx::types::JsonValue
use std::collections::HashMap;
use std::collections::HashSet;
use std::env;
use std::sync::mpsc::{self, Receiver, Sender};

#[derive(Default, Debug, Clone)]
struct Metric {
    xaxis: String,
    orig_values: Vec<[f64; 2]>,
    values: Vec<[f64; 2]>,
    resampled: Vec<[f64; 2]>,
}

#[derive(Default, Debug, Clone)]
struct Run {
    params: HashMap<String, String>,
    metrics: HashMap<String, Metric>,
}

#[derive(Default, Debug, Clone)]
struct Runs {
    runs: HashMap<String, Run>,
    active_runs: Vec<String>, // filtered_runs: HashMap<String, Run>,
}

#[derive(Default, Debug, Clone)]
struct GuiParams {
    n_average: usize,
    max_n: usize,
    param_filters: HashMap<String, HashSet<String>>,
    metric_filters: HashSet<String>,
}

#[derive(PartialEq, Eq)]
enum DataStatus {
    Waiting,
    FirstDataArrived,
    FirstDataProcessed,
    FirstDataPlotted,
}

struct GuiRuns {
    runs: Runs,
    dirty: bool,
    db_train_runs_sender: Sender<Vec<String>>,
    db_reciever: Receiver<HashMap<String, Run>>,
    recomputed_reciever: Receiver<HashMap<String, Run>>,
    dirty_sender: Sender<(GuiParams, HashMap<String, Run>)>,
    initialized: bool,
    data_status: DataStatus,
    gui_params: GuiParams,
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
                        [didx * idx as f64, mean_value]
                    })
                    .collect()
            } else {
                metric.orig_values.clone()
            };
            let fvalues: Vec<[f64; 2]> = (0..metric.values.len())
                .map(|orig_idx| {
                    let window = if gui_params.n_average > 0 {
                        -(gui_params.n_average.min(orig_idx) as i32)
                            ..=(gui_params.n_average.min(metric.values.len() - 1 - orig_idx) as i32)
                    } else {
                        0..=0
                    };
                    let vals: Vec<f64> = window
                        .map(|sub_idx| {
                            let idx = orig_idx as i32 + sub_idx;
                            metric.values[idx as usize][1]
                        })
                        .sorted_by(|a, b| a.partial_cmp(b).unwrap())
                        .collect();
                    let mean_val = vals[vals.len() / 2];
                    [metric.values[orig_idx][0], mean_val]
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
        if self.dirty {
            self.runs.active_runs = get_train_ids_from_filter(&self.runs.runs, &self.gui_params);
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
            self.runs.runs = new_runs;
            if self.data_status == DataStatus::FirstDataArrived && self.runs.runs.len() > 0 {
                self.data_status = DataStatus::FirstDataProcessed;
            }
        }
        if let Ok(new_runs) = self.db_reciever.try_recv() {
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
            .map(|run| run.params.get("ensemble_id").unwrap())
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
                let ensemble_id = run.params.get("ensemble_id").unwrap();
                (train_id.clone(), *ensemble_colors.get(ensemble_id).unwrap())
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
            .active_runs
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
            .width_range(100.0..=300.0)
            .show(ctx, |ui| {
                self.render_parameters(ui, param_values, filtered_values, ctx);
            });
        egui::SidePanel::right("Metrics")
            .resizable(true)
            .default_width(300.0)
            .width_range(100.0..=300.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.vertical(|ui| {
                        for metric_name in &metric_names {
                            let active_filter =
                                self.gui_params.metric_filters.contains(metric_name);
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
            });
        egui::CentralPanel::default().show(ctx, |ui| {
            ctx.request_repaint();
            self.render_plots(ui, metric_names, run_ensemble_color);
            // });
            // });
            // })
        });
        self.initialized = true;
    }
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
                for param_name in param_values.keys().sorted() {
                    ui.separator();
                    ui.label(param_name);
                    ui.horizontal_wrapped(|ui| {
                        for value in param_values.get(param_name).unwrap().iter().sorted() {
                            let active_filter = self
                                .gui_params
                                .param_filters
                                .get(param_name)
                                .unwrap()
                                .contains(value);
                            let filtered_runs_contains =
                                if let Some(values) = filtered_values.get(param_name) {
                                    values.contains(value)
                                } else {
                                    false
                                };
                            let color = if filtered_runs_contains {
                                egui::Color32::LIGHT_GREEN
                            } else {
                                ctx.style().visuals.widgets.inactive.bg_fill
                            };
                            if ui
                                .add(
                                    egui::Button::new(value)
                                        .stroke(egui::Stroke::new(1.0, color))
                                        .selected(active_filter),
                                )
                                .clicked()
                            {
                                if self
                                    .gui_params
                                    .param_filters
                                    .get(param_name)
                                    .unwrap()
                                    .contains(value)
                                {
                                    self.gui_params
                                        .param_filters
                                        .get_mut(param_name)
                                        .unwrap()
                                        .remove(value);
                                } else {
                                    self.gui_params
                                        .param_filters
                                        .get_mut(param_name)
                                        .unwrap()
                                        .insert(value.clone());
                                }
                                self.dirty = true;
                            }
                        }
                    });
                }
            });
        });
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
            ui.available_width()
        } else {
            ui.available_width() / 2.1
        };
        let plot_height = if filtered_metric_names.len() <= 2 {
            ui.available_height() / 2.1
        } else {
            ui.available_height() / 3.1
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
                        .link_axis(
                            **metric_name_axis_id.get(&metric_name).unwrap(),
                            true,
                            false,
                        )
                        .link_cursor(**metric_name_axis_id.get(&metric_name).unwrap(), true, true),
                )
            })
            .collect();
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                for (metric_name, plot) in plots.into_iter().sorted_by_key(|(k, _v)| k.clone()) {
                    ui.allocate_ui(egui::Vec2::from([plot_width, plot_height]), |ui| {
                        ui.vertical_centered(|ui| {
                            ui.label(&metric_name);
                            plot.show(ui, |plot_ui| {
                                for (run_id, run) in
                                    self.runs.active_runs.iter().sorted().map(|train_id| {
                                        (train_id, self.runs.runs.get(train_id).unwrap())
                                    })
                                {
                                    if let Some(metric) = run.metrics.get(&metric_name) {
                                        if self.gui_params.n_average > 1 {
                                            plot_ui.line(
                                                Line::new(PlotPoints::from(metric.values.clone()))
                                                    .name(run.params.get("ensemble_id").unwrap())
                                                    .stroke(Stroke::new(
                                                        1.0,
                                                        run_ensemble_color
                                                            .get(run_id)
                                                            .unwrap()
                                                            .gamma_multiply(0.6),
                                                    )),
                                            );
                                        }

                                        plot_ui.line(
                                            Line::new(PlotPoints::from(metric.resampled.clone()))
                                                .name(run.params.get("ensemble_id").unwrap())
                                                .stroke(Stroke::new(
                                                    2.0,
                                                    *run_ensemble_color.get(run_id).unwrap(),
                                                )),
                                        );
                                        if self.data_status == DataStatus::FirstDataProcessed {
                                            println!("auto bounds");
                                            plot_ui.set_auto_bounds(egui::Vec2b::new(true, true));
                                            // self.data_status = DataStatus::FirstDataProcessed;
                                        }
                                    }
                                }
                            })
                        });
                        // });
                    });
                }
            })
        });
        if self.data_status == DataStatus::FirstDataProcessed {
            // plot_ui.set_auto_bounds(egui::Vec2b::new(true, true));
            self.data_status = DataStatus::FirstDataPlotted;
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
) -> Result<(), sqlx::Error> {
    // Query to get the table structure
    // TODO: WHERE train_id not in runs.keys()
    let run_rows = sqlx::query(
        r#"
        SELECT * FROM runs ORDER BY train_id
        "#,
    )
    .fetch_all(pool)
    .await?;
    // Print each column's details
    // let mut runs: HashMap<String, Run> = HashMap::new();
    for (train_id, params) in &run_rows
        .into_iter()
        .group_by(|row| row.get::<String, _>("train_id"))
    {
        let params: HashMap<_, _> = params
            .map(|row| {
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
                },
            );
        }
    }

    if train_ids.len() > 0 {
        let q = format!(
            r#"
        SELECT * FROM metrics WHERE train_id IN ({}) AND created_at > $1 ORDER BY train_id, variable, xaxis, x
        "#,
            train_ids
                .iter()
                .map(|train_id| format!("'{}'", train_id))
                .join(","),
        );
        println!("{}", q);
        let metric_rows = sqlx::query(q.as_str())
            .bind(*last_timestamp)
            .fetch_all(pool)
            .await?;
        for (train_id, run_metric_rows) in &metric_rows
            .into_iter()
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
                let orig_values: Vec<_> = rows
                    .iter()
                    .filter_map(|row| {
                        // println!("{:?}", row.try_get::<i32, _>("id"));
                        if let Ok(x) = row.try_get::<f64, _>("x") {
                            if let Ok(value) = row.try_get::<f64, _>("value") {
                                return Some([x, value]);
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
                        },
                    );
                } else {
                    run.metrics
                        .get_mut(&variable)
                        .unwrap()
                        .orig_values
                        .extend(orig_values);
                }
            }
        }
    }
    Ok(())
    // Ok(Runs {
    //     filtered_runs: runs.clone(),
    //     runs,
    // })
}
// #[tokio::main(flavor = "current_thread")]
fn main() -> Result<(), sqlx::Error> {
    // Load environment variables
    // dotenv::dotenv().ok();
    let (tx, rx) = mpsc::channel();
    let (tx_gui_dirty, rx_gui_dirty) = mpsc::channel();
    let (tx_gui_recomputed, rx_gui_recomputed) = mpsc::channel();
    let (tx_db_filters, rx_db_filters) = mpsc::channel();

    std::thread::spawn(move || loop {
        if let Ok((gui_params, mut runs)) = rx_gui_dirty.recv() {
            recompute(&mut runs, &gui_params);
            tx_gui_recomputed
                .send(runs)
                .expect("Failed to send recomputed runs.");
        }
    });
    // Connect to the database

    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
            let pool = PgPoolOptions::new()
                .max_connections(5)
                .connect(&database_url)
                .await
                .expect("Can't connect to database");
            let mut train_ids = Vec::new();
            let mut last_timestamp = NaiveDateTime::from_timestamp_millis(0).unwrap();
            let mut runs: HashMap<String, Run> = Default::default();
            loop {
                // Fetch data from database
                println!("Getting state...");
                get_state_new(&pool, &mut runs, &train_ids, &mut last_timestamp)
                    .await
                    .expect("Get state:");
                // println!("{:?}", runs);
                println!("Done.");
                // Send data to UI thread
                // if let Ok(runs) = runs e
                tx.send(runs.clone()).expect("Failed to send data");
                // }
                // Wait some time before fetching again
                for _ in 0..100 {
                    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                    if let Ok(new_train_ids) = rx_db_filters.try_recv() {
                        train_ids = new_train_ids;
                        last_timestamp = NaiveDateTime::from_timestamp_millis(0).unwrap();
                        runs = HashMap::new();
                        break;
                    }
                }
            }
        });
    });

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([350.0, 200.0]),
        ..Default::default()
    };
    let _ = eframe::run_native(
        "Visualizer",
        options,
        Box::new(|_cc| {
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
                    n_average: 1,
                },
            })
        }),
    );

    Ok(())
}
