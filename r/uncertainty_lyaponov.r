library(patchwork)
library(ggplot2)
library(reticulate)
uncertainty <-
  read.csv("~/projects/equivariant-posteriors/experiments/lyaponov/uncertainty_lyaponov.csv")
xy_MI <-
  ggplot2::ggplot(uncertainty, aes(x = x, y = y)) + geom_point(aes(color =
                                                                     MI),
                                                               size=0.6,
                                                               alpha=0.5) + scale_color_gradientn(colours = rainbow(5))
xy_H <-
  ggplot2::ggplot(uncertainty, aes(x = x, y = y)) + geom_point(aes(color =
                                                                     H), 
                                                               size=0.6,
                                                               alpha=0.5) + scale_color_gradientn(colours = rainbow(5))

xy_A <-
  ggplot2::ggplot(uncertainty, aes(x = x, y = y)) + geom_point(aes(color =
                                                                     H-MI), 
                                                               size=0.6,
                                                               alpha=0.5) + scale_color_gradientn(colours = rainbow(5))
#MIH <-
#  ggplot(uncertainty, aes(x = H, y = MI)) +
#  geom_density_2d_filled() +
#  scale_x_continuous(trans = "log2") +
#  scale_y_continuous(trans = "log2")
#MIH_points <- ggplot(uncertainty, aes(x = H, y = MI)) +
#  geom_point() +
#  scale_x_continuous(trans = "log2") +
#  scale_y_continuous(trans = "log2")
##xy_MI + xy_H + MIH + MIH_points + plot_layout(ncol=2)
xy_MI + xy_H + xy_A + plot_layout(ncol=3)

