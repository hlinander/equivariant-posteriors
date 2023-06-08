library(patchwork)
library(ggplot2)
library(Hmisc)
uncertainty <-
  read.csv("~/projects/equivariant-posteriors/experiments/lyaponov/mnist_test_uncertainty.csv")
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
xy_L <-
  ggplot2::ggplot(uncertainty, aes(x = lx, y = ly)) + geom_point(aes(color =
                                                                     FTLE), 
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
full <- xy_MI + xy_H + xy_L# + plot_layout(ncol=3)
#lim <- xlim(xmin, xmax) + ylim(ymin, ymax)
xmin <- -2.7
xmax <- 8
ymin <- -13
ymax <- -7.0
#xmin <- -13
#xmax <- -7
#ymin <- -7.5
#ymax <- -2.5

mi_b <- xy_MI + xlim(xmin, xmax) + ylim(ymin, ymax) 
h_b <-  xy_H + xlim(xmin, xmax) + ylim(ymin, ymax)
FTLE_b <- xy_L + xlim(xmin, xmax) + ylim(ymin, ymax)

#+ plot_layout(ncol=3)
(xy_MI + geom_rect(aes(xmin=xmin, xmax=xmax, ymin=ymin,ymax=ymax), fill=NA, color="black")) + 
  xy_H  + geom_rect(aes(xmin=xmin, xmax=xmax, ymin=ymin,ymax=ymax), fill=NA, color="black") + 
  xy_L  + geom_rect(aes(xmin=xmin, xmax=xmax, ymin=ymin,ymax=ymax), fill=NA, color="black") + 
  mi_b + h_b + FTLE_b + plot_layout(ncol=3)
#breakout
#library(dplyr)
#uncertainty2 <- dplyr::filter(uncertainty, (lx-x)^2 + (ly-y)^2 > 0.0000000001)
#ggplot2::ggplot(uncertainty2, aes(x=x, y=y)) + geom_point()
(ftle_mi_exp_lambda <- ggplot2::ggplot(uncertainty, aes(x=exp(FTLE), y=MI))
  + geom_point(alpha=0.1, size=0.2, color="blue") 
  #+ geom_density_2d()
  + stat_summary_bin(fun.data="mean_sdl", bins=60)
  
  )
(ftle_mi_lambda <- ggplot2::ggplot(uncertainty, aes(x=FTLE, y=MI))
  + geom_point(alpha=0.1, size=0.2, color="blue") 
  #+ geom_density_2d()
  + stat_summary_bin(fun.data="mean_sdl", bins=60)
  
)
(ftle_mi_exp_lambda_log <- ggplot2::ggplot(uncertainty, aes(x=exp(FTLE), y=MI))
  + geom_point(alpha=0.1, size=0.2, color="blue") 
  #+ geom_density_2d()
  + stat_summary_bin(fun.y="mean", bins=60)
  + scale_y_continuous(trans="log2")
)
(ftle_mi_lambda_log <- ggplot2::ggplot(uncertainty, aes(x=FTLE, y=MI))
  + geom_point(alpha=0.1, size=0.2, color="blue") 
  #+ geom_density_2d()
  + stat_summary_bin(fun.y="mean", bins=60)
  + scale_y_continuous(trans="log2")
)
  #scale_x_continuous(trans = "log2") #+ #+ geom_density_2d_filled() + scale_x_continuous(trans = "log2") + scale_y_continuous(trans = "log2")
  #scale_y_continuous(trans = "log2")
#ftle_h <- (ggplot2::ggplot(uncertainty, aes(x=FTLE, y=H-MI)) 
#  + stat_summary_bin(fun.data="mean_sdl", bins=50))
  #scale_x_continuous(trans = "log2") #+ #+ geom_density_2d_filled() + scale_x_continuous(trans = "log2") + scale_y_continuous(trans = "log2")
  #scale_y_continuous(trans = "log2")
(
  (ftle_mi_lambda_log)
+ (ftle_mi_exp_lambda_log)
+ ftle_mi_lambda 
+ ftle_mi_exp_lambda 
+ plot_layout(ncol=2)
)
