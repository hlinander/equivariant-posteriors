library(ggplot2)
library(dplyr)
library(tikzDevice)

df1 <- read.csv("./alvis2/al_a9257be/conv_calibrated_uq_tn.csv")
df2 <- read.csv("./alvis2/al_8ceba8d/conv_calibrated_uq.csv")
df3 <- read.csv("./alvis2/al_52feb2d/conv_predictive_entropy.csv")
df4 <- read.csv("./alvis2/al_3be58db/mlp_calibrated_uq_tn.csv")
df5 <- read.csv("./alvis2/al_7635944/conv_mutual_information.csv")
df6 <- read.csv("./alvis2/al_87d4450/mlp_calibrated_uq.csv")
df7 <- read.csv("./alvis2/al_eecb9a7/mlp_mutual_information.csv")
df8 <- read.csv("./alvis2/al_db45db6/mlp_predictive_entropy.csv")
df9 <- read.csv("./alvis2/al_8ac1e0b/conv_random.csv")
dfa <- read.csv("./alvis2/al_dd352a5/mlp_random.csv")
df <- rbind(df1,df2,df3,df4,df5,df6,df7,df8,df9,dfa)
df_conv_mlp <- df %>% filter(aquisition=="calibrated_uq" | aquisition=="mutual_information" | aquisition=="predictive_entropy")
p<-(ggplot(df, aes(x=n_aquired, y=value, group=interaction(model, aquisition), color=aquisition)) + geom_line(aes(linetype=model)) + 
      facet_grid(rows=vars(metric),scales="free_y") +
      theme_minimal())
ggsave("./experiments/looking_at_the_posterior/al_cifar_balanced.pdf", p)
print(p)

df_cal<-read.csv("./alvis2/al_8ceba8d/uq_calibration_step_014_calibrated_uq.csv")
df_aq <-read.csv("./alvis2/al_8ceba8d/uq_aquired_step_014_calibrated_uq.csv")
df_pool <-read.csv("./alvis2/al_8ceba8d/uq_pool_step_014_calibrated_uq.csv")
min_samples_mean <- function(z) {
  if (length(z) >= 3) {
    return(mean(z))
  } else {
    return(NA)  # return NA for bins with less than 10 samples
  }
}


p <- (ggplot() +
    stat_summary_2d(data=df_cal, aes(x=H, y=MI, z=accuracy, fill=after_stat(value)), fun=min_samples_mean, bins=15) +
    geom_point(data=sample_n(df_pool, 1000) %>% filter(target < 11), aes(x=H, y=MI), color="red", shape=1, alpha=0.5) +
      geom_point(data=df_aq, aes(x=H, y=MI), color="black", fill="green", shape=21, size=2, stroke=0.5) +
    scale_fill_viridis_c() +
      labs(fill="Accuracy") + 
      xlab("H, Predictive uncertainty") +
      ylab("I, Epistemic uncertainty") +
    theme_minimal(base_size=7) #+ 
  #scale_x_continuous(trans = log_trans()) +
  #scale_y_continuous(trans = log_trans())
)
print(p)
ggsave("./experiments/looking_at_the_posterior/al_cifar_balanced_aquisition.pdf", p, width=6.17*0.9*0.6, height=6.17*0.9*0.6*2.5/4)
tikz("./experiments/looking_at_the_posterior/al_cifar_balanced_aquisition.tex", width=6.17*0.9*0.45, height=6.17*0.9*0.6*2.5/4, verbose=FALSE, pointsize = 12)
print(p)
dev.off()

aquisition_map = c("calibrated_uq"="Calibrated UQ", "mutual_information"="Mutual information", "predictive_entropy"="Predictive entropy")
df_conv_mlp <- df %>% filter(metric=="accuracy" & (aquisition=="calibrated_uq" | aquisition=="mutual_information" | aquisition=="predictive_entropy"))
df_conv_mlp$humanaq = aquisition_map[df_conv_mlp$aquisition]
p<-(ggplot(df_conv_mlp, aes(x=n_aquired, y=value, group=interaction(model, humanaq), color=humanaq)) + geom_line(aes(linetype=model)) + 
      geom_vline(xintercept = 14*50, linetype="dashed", color = "black", linewidth=1.0) +
      #facet_grid(rows=vars(metric),scales="free_y") +
      labs(color="Aquisition method", linetype="Model") + 
      xlab("Aquired samples from pool") +
      ylab("Validation accuracy") +
      #theme(axis.text.y.right = element_blank(),axis.ticks.y.right = element_blank()))
      theme_minimal(base_size = 7))
ggsave("./experiments/looking_at_the_posterior/al_cifar_balanced.pdf", p)
print(p)
tikz("./experiments/looking_at_the_posterior/al_cifar_balanced.tex", width=6.17*0.9*0.5, height=6.17*0.9*0.6*2.5/4, verbose=FALSE, pointsize = 12)
print(p)
dev.off()
