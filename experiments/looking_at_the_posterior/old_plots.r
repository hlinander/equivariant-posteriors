library(ggplot2)
library(ggExtra)
library(dplyr)
library(matlab)
library(patchwork)
library(ggpubr)

df <- read.csv("./experiments/looking_at_the_posterior/old_paper_plot_data/data.csv")
unique(df$alpha)

p <- (ggplot(df %>% filter(method=="ensemble", alpha==0.09), aes(x=H, y=MI, z=accuracy)) + stat_summary_2d(fun="mean", bins=50, aes(fill=..value..)) + 
  #geom_density2d(contour_var = "ndensity", bins=num_levels) +
  geom_tile() +
  geom_abline(color="red", linetype="dashed", linewidth=2) +
  #scale_fill_viridis_c(name="Avg. Accuracy", direction = 1) + 
  scale_fill_viridis_c(name="Accuracy", direction = 1,option = "H", alpha=1) + 
  scale_x_log10(limits=c(1e-3, 3)) +
  scale_y_log10(limits=c(1e-3, 3)) +
  theme_minimal(base_size = 7) +
  labs(x="H, Predictive uncertainty", y="MI, Epistemic uncertainty", z="Accuracy") + 
    facet_grid(~model))
print(p)
