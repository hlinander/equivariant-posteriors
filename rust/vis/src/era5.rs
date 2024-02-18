// meteorological_data.rs

#[derive(Debug, Clone)]
pub struct MeteorologicalData<'a> {
    pub surface: Category<'a>,
    pub upper: Category<'a>,
}

#[derive(Debug, Clone)]
pub struct Category<'a> {
    pub names: Vec<&'a str>,
    pub long_names: Vec<&'a str>,
    pub units: Vec<&'a str>,
    pub levels: Vec<f32>,
    pub level_units: &'a str,
    pub level_name: &'a str,
}

pub fn era5_meta() -> MeteorologicalData<'static> {
    MeteorologicalData {
        surface: Category {
            names: vec!["msl", "u10", "v10", "t2m"],
            long_names: vec![
                "Mean sea level pressure",
                "10 metre U wind component",
                "10 metre V wind component",
                "2 metre temperature",
            ],
            units: vec!["Pa", "m s**-1", "m s**-1", "K"],
            levels: vec![],
            level_units: "hPa",
            level_name: "Geopotential height",
        },
        upper: Category {
            names: vec!["z", "q", "t", "u", "v"],
            long_names: vec![
                "Geopotential",
                "Specific humidity",
                "Temperature",
                "U component of wind",
                "V component of wind",
            ],
            units: vec!["m**2 s**-2", "kg kg**-1", "K", "m s**-1", "m s**-1"],
            levels: vec![
                1000., 925., 850., 700., 600., 500., 400., 300., 250., 200., 150., 100., 50.,
            ],
            level_units: "hPa",
            level_name: "Geopotential height",
        },
    }
}
