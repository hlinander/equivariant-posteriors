use eframe::egui;
use egui::epaint;
use egui_plot::{Legend, PlotPoints};
use egui_plot::{Line, Plot};
use itertools::Itertools;
use sqlx::postgres::PgColumn;
use sqlx::postgres::PgPoolOptions;
use sqlx::postgres::PgRow;
use sqlx::Column;
use sqlx::Row;
use std::collections::HashMap;
use std::collections::HashSet;
use std::env;
use std::rc::Rc;
use std::sync::mpsc::{self, Receiver};
use tokio::main;

#[derive(Default, Debug, Clone)]
struct Run {
    params: HashMap<String, String>,
    metrics: HashMap<String, Vec<(i32, f64)>>,
}

#[derive(Default, Debug)]
struct Runs {
    runs: HashMap<String, Run>,
}

struct GuiRuns {
    runs: Runs,
    reciever: Receiver<Runs>,
    initialized: bool,
    n_average: usize,
    filters: HashMap<String, HashSet<String>>,
}

impl eframe::App for GuiRuns {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        ctx.set_zoom_factor(2.0);
        if let Ok(new_runs) = self.reciever.try_recv() {
            self.runs = new_runs;
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
        let param_values = get_parameter_values(&self.runs.runs);
        for param_name in param_values.keys() {
            if !self.filters.contains_key(param_name) {
                self.filters.insert(param_name.clone(), HashSet::new());
            }
        }
        let filtered_runs: HashMap<String, Run> = self
            .runs
            .runs
            .iter()
            .filter(|run| {
                for (param_name, values) in self.filters.iter().filter(|(_, vs)| !vs.is_empty()) {
                    if let Some(run_value) = run.1.params.get(param_name) {
                        if !values.contains(run_value) {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
                true
            })
            .map(|(p, v)| (p.clone(), v.clone()))
            .collect();
        // for run in &filtered_runs {
        //     println!("{:?}", run.1.params.get("epochs"))
        // }
        let filtered_values = get_parameter_values(&filtered_runs);
        let metric_names: Vec<String> = filtered_runs
            .values()
            .map(|run| run.metrics.keys().cloned())
            .flatten()
            .unique()
            .collect();

        let plots: HashMap<_, _> = metric_names
            .into_iter()
            .map(|metric_name| {
                (
                    metric_name.clone(),
                    Plot::new(metric_name)
                        .legend(Legend::default())
                        .width(500.0)
                        .height(500.0),
                )
            })
            .collect();
        egui::SidePanel::left("Controls")
            .resizable(true)
            .default_width(200.0)
            .width_range(100.0..=300.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.vertical(|ui| {
                        ui.add(egui::Slider::new(&mut self.n_average, 1usize..=20usize));
                        for param_name in param_values.keys().sorted() {
                            ui.separator();
                            ui.label(param_name);
                            ui.horizontal_wrapped(|ui| {
                                for value in param_values.get(param_name).unwrap().iter().sorted() {
                                    let active_filter =
                                        self.filters.get(param_name).unwrap().contains(value);
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
                                        if self.filters.get(param_name).unwrap().contains(value) {
                                            self.filters.get_mut(param_name).unwrap().remove(value);
                                        } else {
                                            self.filters
                                                .get_mut(param_name)
                                                .unwrap()
                                                .insert(value.clone());
                                        }
                                    }
                                }
                            });
                        }
                    });
                });
            });
        egui::CentralPanel::default().show(ctx, |ui| {
            ctx.request_repaint();
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    for (metric_name, plot) in plots.into_iter().sorted_by_key(|(k, v)| k.clone()) {
                        ui.allocate_ui(egui::Vec2::from([500.0, 500.0]), |ui| {
                            ui.vertical_centered(|ui| {
                                ui.label(&metric_name);
                                plot.show(ui, |plot_ui| {
                                    for (run_id, run) in
                                        filtered_runs.iter().sorted_by_key(|(k, _v)| *k)
                                    {
                                        if let Some(metric) = run.metrics.get(&metric_name) {
                                            let mut fvalues: Vec<[f64; 2]> = metric
                                                .iter()
                                                .map(|(i, v)| [*i as f64, *v as f64])
                                                .collect::<Vec<_>>();
                                            let last_val = fvalues.last().unwrap().clone();
                                            // let first_val = fvalues.first().unwrap().clone();
                                            // let pre = vec![first_val; self.n_average / 2];
                                            // fvalues = [pre, fvalues].concat();
                                            fvalues.extend(
                                                std::iter::repeat(last_val).take(self.n_average),
                                            );
                                            let fvalues: Vec<[f64; 2]> = fvalues
                                                .windows(self.n_average)
                                                .map(|window| {
                                                    let sum: f64 =
                                                        window.iter().map(|vals| vals[1]).sum();
                                                    [window[0][0], sum / window.len() as f64]
                                                })
                                                .collect();
                                            plot_ui.line(
                                                Line::new(PlotPoints::from(fvalues))
                                                    .name(run.params.get("ensemble_id").unwrap())
                                                    .color(
                                                        *run_ensemble_color.get(run_id).unwrap(),
                                                    ),
                                            );
                                            if !self.initialized {
                                                plot_ui
                                                    .set_auto_bounds(egui::Vec2b::new(true, true));
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
            // });
            // });
            // })
        });
        self.initialized = true;
    }
}

fn get_parameter_values(lruns: &HashMap<String, Run>) -> HashMap<String, HashSet<String>> {
    let mut param_values: HashMap<String, HashSet<String>> = lruns
        .values()
        .map(|run| run.params.keys().cloned())
        .flatten()
        .unique()
        .map(|param_name| (param_name, HashSet::new()))
        .collect();
    for run in lruns.values() {
        for (k, v) in &run.params {
            let values = param_values.get_mut(k).unwrap();
            values.insert(v.clone());
        }
    }
    param_values
}

async fn get_state_new(pool: &sqlx::postgres::PgPool) -> Result<Runs, sqlx::Error> {
    // Query to get the table structure
    let run_rows = sqlx::query(
        r#"
        SELECT * FROM runs ORDER BY train_id
        "#,
    )
    .fetch_all(pool)
    .await?;
    // Print each column's details
    let mut runs: HashMap<String, Run> = HashMap::new();
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
        runs.insert(
            train_id,
            Run {
                params,
                metrics: HashMap::new(),
            },
        );
    }

    let metric_rows = sqlx::query(
        r#"
        SELECT * FROM metrics ORDER BY train_id, variable, epoch
        "#,
    )
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
            let values: Vec<_> = value_rows
                .map(|row| (row.get::<i32, _>("epoch"), row.get::<f64, _>("value")))
                .collect();
            run.metrics.insert(variable, values);
        }
    }
    Ok(Runs { runs })
}
// #[tokio::main(flavor = "current_thread")]
fn main() -> Result<(), sqlx::Error> {
    // Load environment variables
    // dotenv::dotenv().ok();
    let (tx, rx) = mpsc::channel();

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

            loop {
                // Fetch data from database
                println!("Getting state...");
                let runs = get_state_new(&pool).await.expect("Get state:");
                println!("Done.");
                // Send data to UI thread
                // if let Ok(runs) = runs {
                tx.send(runs).expect("Failed to send data");
                // }
                // Wait some time before fetching again
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            }
        });
    });

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([350.0, 200.0]),
        ..Default::default()
    };
    let _ = eframe::run_native(
        "My egui App with a plot",
        options,
        Box::new(|_cc| {
            Box::<GuiRuns>::new(GuiRuns {
                runs: Default::default(),
                reciever: rx,
                initialized: false,
                n_average: 1,
                filters: HashMap::new(),
            })
        }),
    );

    Ok(())
}
