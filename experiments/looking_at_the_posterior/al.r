library(ggplot2)
library(RPostgreSQL)

drv <- dbDriver("PostgreSQL")
con <-
  dbConnect(
    drv,
    dbname = "equiv",
    host = "localhost",
    port = "5432",
    user = "postgres",
    password = "postgres"
  )

# Fetch data from my_table
query <- "SELECT * FROM active_learning;"
df <- dbGetQuery(con, query)

# Close the database connection
dbDisconnect(con)
df1 <- read.csv("./alvis/al_33fd470/calibrated_uncertainty.csv")
df2 <- read.csv("./alvis/al_33fd470/random.csv")
df <- rbind(df1, df2)
ggplot(df, aes(x=fraction, y=value, group=model, color=model)) + geom_line() + facet_wrap(~metric,scales="free_y")
df <- read.csv("./experiments/looking_at_the_posterior/al_794843e/uq_calibration.csv")
p <- (ggplot(df, aes(x = fraction, y = value, group = model, color=model)) + 
        geom_line() + 
        facet_wrap( ~metric, scales="free_y") + 
        theme_minimal(base_size = 24) +
        xlim(0, 1)
print(p)

df_cal<-read.csv("./alvis/al_28ba7eb/uq_calibration_step_006_calibrated_uncertainty.csv")
df_aq <-read.csv("./alvis/al_28ba7eb/uq_aquired_step_006_calibrated_uncertainty.csv")
df_pool <-read.csv("./alvis/al_28ba7eb/uq_pool_step_006_calibrated_uncertainty.csv")
min_samples_mean <- function(z) {
  if (length(z) >= 10) {
    return(mean(z))
  } else {
    return(NA)  # return NA for bins with less than 10 samples
  }
}

(ggplot() +
stat_summary_2d(data=df_cal, aes(x=H, y=MI, z=accuracy, fill=..value..), fun="mean", bins=50) +
   geom_point(data=df_aq, aes(x=H, y=MI), color="red", fill="green", shape=19, size=5) +
    geom_point(data=df_pool, aes(x=H, y=MI), color="red", shape=2) +
    scale_fill_viridis_c() +
    theme_minimal(base_size=40)
  )


(ggplot(df, aes(x=H, y=MI, z=accuracy)) +
    stat_summary_2d(aes(fill=..value..), fun="mean", bins=20) +
    scale_fill_viridis_c() +
    theme_minimal(base_size=40)
)

df<-read.csv("./alvis/al_28ba7eb/uq_pool_step_003_calibrated_uncertainty.csv")
(ggplot(df, aes(x=H, y=MI, z=accuracy)) +
    stat_summary_2d(aes(fill=..value..), fun="mean", bins=20) +
    scale_fill_viridis_c() +
    theme_minimal(base_size=40)
)
