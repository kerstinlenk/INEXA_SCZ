##### Analyse experiments #####
##### Set up files and random seed ####

# set.seed(450)
set.seed(451)
# set.seed(452)

## Import csv
(X_orig = read.csv("/home/flo/Projects/Astrocyte_Project/Clean/Data/allresults_in_ex_experiments.csv"))

## Only keep columns of interest
(X = X_orig[,2:length((X_orig[1,]))])
X$in. = -X$in.

## Look at the structure of the data
str(X)

## Re-configure as categorical
X$net = as.factor(X$net)
X$con = as.factor(X$con)

# Structure of data after reconfiguration
str(X)
(data = X)

##### Interaction and exploratory analysis ##### 
par(mfrow=c(3,1))

## Interaction of Omega_G with Omega_F: 
# Does the slope when going between values of omega_G depend on Omega_F?
with(data, interaction.plot(x.factor = interaction(net,con),
                            trace.factor = interaction(in.,ex)
                            ,
                            response = BurstD0,
                            col = rep(c("red", "blue","green"), times = 4)))
with(data, interaction.plot(x.factor = interaction(param_G,param_F),
                            trace.factor = interaction(net,con),
                            response = BurstD2,
                            col = rep(c("red", "blue","green"), times = 4)))
with(data, interaction.plot(x.factor = interaction(param_G,param_F),
                            trace.factor = interaction(net,con),
                            response = MeanAA,
                            col = rep(c("red", "blue","green"), times = 4)))

# the plots indicate that the slopes are not exactly parallel, but relatively close to parallel.

## Interaction of Omega_G with Omega_F: 
# Does the slope when going between values of omega_F depend on Omega_G?
with(data, interaction.plot(x.factor = interaction(param_F,param_G),
                            trace.factor = interaction(net,con),
                            response = BurstD0,
                            col = rep(c("red", "blue","green"), times = 4)))
with(data, interaction.plot(x.factor = interaction(param_F,param_G),
                            trace.factor = interaction(net,con),
                            response = BurstD2,
                            col = rep(c("red", "blue","green"), times = 4)))
with(data, interaction.plot(x.factor = interaction(param_F,param_G),
                            trace.factor = interaction(net,con),
                            response = MeanAA,
                            col = rep(c("red", "blue","green"), times = 4)))

# slopes of transitions 2.2. --> 4.2 and 2.1 --> 4.1 seem to depend strongly on network / connectivity for 
# BurstD0-3. In BurstD4-5 we get a major outlier for the net-con configuration (2,0) and (o_F,o_G)=(2,2).

# boxplots ####
par(mfrow=c(3,1))
# par(mfrow=c(1,1))
boxplot(BurstD0~interaction(in.,ex), data=data)
boxplot(BurstD2~interaction(in.,ex), data=data)
boxplot(MeanNrActivationsPerAstrocyte~interaction(in.,ex), data=data)

par(mfrow=c(3,1))
boxplot(BurstD0~interaction(in.,ex), data=data)
boxplot(BurstD2~interaction(in.,ex), data=data)
boxplot(MeanNrActivationsPerAstrocyte~interaction(in.,ex), data=data)
##### Analysis using mixed effects model ####

# Required packages
library(lmerTest)
library(multcomp)
library(stargazer)

options(contrasts = c("contr.treatment", "contr.poly"))

(X_scaled_non_centered = cbind(subset(X,select = c(net,con)),
                               as.data.frame(scale(subset(X,select = -c(net,con)), center = FALSE, scale = apply(subset(X,select = -c(net,con)), 2, sd, na.rm = TRUE)))))

fit_0 = lmer(BurstD0                       ~ in. + ex + in.:ex + I(in.^2) + I(ex^2) + (1 | net:con), data=X_scaled_non_centered)
fit_1 = lmer(BurstD2                       ~ in. + ex + in.:ex + I(in.^2) + I(ex^2) + (1 | net:con), data=X_scaled_non_centered)
fit_2 = lmer(MeanNrActivationsPerAstrocyte ~ in. + ex + in.:ex + I(in.^2) + I(ex^2) + (1 | net:con), data=X_scaled_non_centered)


# diagnostic TA plots, qq-plots ####
par(mfrow=c(1,1))
plot(fit_0)
# spreadLevelPlot(fit_0)
qqnorm(resid(fit_0), main = "Residuals QQ-Plot")
qqline(resid(fit_0), main = "Residuals QQ-Plot")

qqnorm(ranef(fit_0)$'net:con'[,1], main = "Residuals QQ-Plot")
qqline(ranef(fit_0)$'net:con'[,1], main = "Residuals QQ-Plot")

plot(fit_1)
qqnorm(resid(fit_1), main = "Residuals QQ-Plot")
qqline(resid(fit_1), main = "Residuals QQ-Plot")

qqnorm(ranef(fit_1)$'net:con'[,1], main = "Residuals QQ-Plot")
qqline(ranef(fit_1)$'net:con'[,1], main = "Residuals QQ-Plot")

par(mfrow=c(2,1))
plot(fit_2)
qqnorm(resid(fit_2), main = "Residuals QQ-Plot")
qqline(resid(fit_2), main = "Residuals QQ-Plot")

qqnorm(ranef(fit_2)$'net:con'[,1], main = "Residuals QQ-Plot")
qqline(ranef(fit_2)$'net:con'[,1], main = "Residuals QQ-Plot")


# # summaries ####
# response_OLD = function(x,y,s){
#   return({s$coefficients[1]+s$coefficients[2]*x+s$coefficients[3]*y+s$coefficients[6]*x*y+s$coefficients[4]*x^2+s$coefficients[5]*y^2}^2)
# }
# X
# (sd_df = apply(X,2,sd))

(sd_df = apply(subset(X, select = -c(net,con)),2,sd))

response_0 = function(x,y,s){
  x_new = 100*x/sd_df["in."]
  y_new = 100*y/sd_df["ex"]
  return(sd_df["BurstD0"]*(s$coefficients[1]
                           + s$coefficients[2]*x_new    
                           + s$coefficients[3]*y_new
                           + s$coefficients[6]*x_new*y_new
                           + s$coefficients[4]*x_new^2
                           + s$coefficients[5]*y_new^2))
}

response_1 = function(x,y,s){
  x_new = 100*x/sd_df["in."]
  y_new = 100*y/sd_df["ex"]
  return(sd_df["BurstD2"]*(s$coefficients[1]
                           + s$coefficients[2]*x_new    
                           + s$coefficients[3]*y_new
                           + s$coefficients[6]*x_new*y_new
                           + s$coefficients[4]*x_new^2
                           + s$coefficients[5]*y_new^2))
}

response_2 = function(x,y,s){
  x_new = 100*x/sd_df["in."]
  y_new = 100*y/sd_df["ex"]
  return(sd_df["MeanNrActivationsPerAstrocyte"]*(s$coefficients[1]
                                                 + s$coefficients[2]*x_new    
                                                 + s$coefficients[3]*y_new
                                                 + s$coefficients[6]*x_new*y_new
                                                 + s$coefficients[4]*x_new^2
                                                 + s$coefficients[5]*y_new^2))
}
(x = seq(1/100*sd_df["in."]*min(X_scaled_non_centered["in."]),1/100*sd_df["in."]*max(X_scaled_non_centered["in."]), length.out = 1e3))
(y = seq(1/100*sd_df["ex"]*min(X_scaled_non_centered["ex"]),1/100*sd_df["ex"]*max(X_scaled_non_centered["ex"]), length.out = 1e3))

(store = c(1/100*sd_df["in."]*min(X_scaled_non_centered["in."]),
           1/100*sd_df["in."]*max(X_scaled_non_centered["in."]),
           1/100*sd_df["ex"]*min(X_scaled_non_centered["ex"]),
           1/100*sd_df["ex"]*max(X_scaled_non_centered["ex"]),
          sd_df["in."],
          sd_df["ex"]
))

write.csv(store,
          file="/home/flo/Projects/Astrocyte_Project/Clean/Data/syn_plot_boundaries.csv")

## fit 0

(s = summary(fit_0))

(store = c(s$coefficients[,1],
           sd_df["BurstD0"]
           ))

write.csv(store,
          file="/home/flo/Projects/Astrocyte_Project/Clean/Data/syn_fits_msr.csv")

# z = matrix(data=NA, nrow = 1e3, ncol = 1e3)
# for (i in c(1:1e3)) {
#   for (j in c(1:1e3)) {
#     z[i,j] = response_0(x[i],y[j],s)
#   }
# }
# 
# z_max = max(z)
# z_min = min(z)
# (z_range = seq(z_min,z_max,length.out = 30))
# (zero_val = length(z_range[z_range<0]))
# 
# col_pal = function(n) {
#   return(c(rep("grey",max(0,times=zero_val-1)),hcl.colors(min(n-zero_val+1,n), "geyser", rev = FALSE)))
# }
# par(mfrow=c(1,1))
# 
# t0 = filled.contour(x,y,z, 
#                     levels = z_range, 
#                     xlab = expression(y["in"]), 
#                     ylab = expression(y["ex"]), 
#                     color.palette = col_pal,
#                     main = "Mean Spike Rate"
# )

(c = confint(fit_0))
(estimates = t(cbind(attr(s$varcor$`net:con`,"stddev"),
                     s$sigma,
                     t(s$coefficients[,1]))))

(tble0 = cbind("Estimates" = estimates,
               "2.5 CI" = c[,1],
               "97.5 CI" =c[,2]
))
tble0_re_scaled_0 = tble0
tble0_re_scaled_0[1,] = sd_df["BurstD0"]*tble0[1,]
tble0_re_scaled_0[2,] = sd_df["BurstD0"]*tble0[2,]
tble0_re_scaled_0[3,] = sd_df["BurstD0"]*tble0[3,]
tble0_re_scaled_0[4,] = sd_df["BurstD0"]/sd_df["in."]*tble0[4,]*100
tble0_re_scaled_0[5,] = sd_df["BurstD0"]/sd_df["ex"]*tble0[5,]*100
tble0_re_scaled_0[6,] = sd_df["BurstD0"]/sd_df["ex"]^2*tble0[6,]*100^2
tble0_re_scaled_0[7,] = sd_df["BurstD0"]/sd_df["ex"]^2*tble0[7,]*100^2
tble0_re_scaled_0[8,] = sd_df["BurstD0"]/(sd_df["in."]*sd_df["ex"])*tble0[8,]*100^2

tble0_re_scaled_0

### fit1

(s = summary(fit_1))

(store = c(s$coefficients[,1],
           sd_df["BurstD2"]
))

write.csv(store,
          file="/home/flo/Projects/Astrocyte_Project/Clean/Data/syn_fits_mbr.csv")

# z = matrix(data=NA, nrow = 1e3, ncol = 1e3)
# for (i in c(1:1e3)) {
#   for (j in c(1:1e3)) {
#     z[i,j] = response_1(x[i],y[j],s)
#   }
# }
# 
# z_max = max(z)
# z_min = min(z)
# (z_range = seq(z_min,z_max,length.out = 30))
# (zero_val = length(z_range[z_range<0]))
# 
# col_pal = function(n) {
#   return(c(rep("grey",max(0,times=zero_val-1)),hcl.colors(min(n-zero_val+1,n), "geyser", rev = FALSE)))
# }
# par(mfrow=c(1,1))
# 
# t1 = filled.contour(x,y,z, 
#                     levels = z_range, 
#                     xlab = expression(y[in]), 
#                     ylab = expression(y_ex), 
#                     color.palette = col_pal,
#                     main = "Mean Burst Rate"
# )


(c = confint(fit_1))
(estimates = t(cbind(attr(s$varcor$`net:con`,"stddev"),
                     s$sigma,
                     t(s$coefficients[,1]))))

(tble1 = cbind("Estimates" = estimates,
               "2.5 CI" = c[,1],
               "97.5 CI" =c[,2]
))

tble1_re_scaled_1 = tble1
tble1_re_scaled_1[1,] = sd_df["BurstD2"]*tble1[1,]
tble1_re_scaled_1[2,] = sd_df["BurstD2"]*tble1[2,]
tble1_re_scaled_1[3,] = sd_df["BurstD2"]*tble1[3,]
tble1_re_scaled_1[4,] = sd_df["BurstD2"]/sd_df["in."]*tble1[4,]*100
tble1_re_scaled_1[5,] = sd_df["BurstD2"]/sd_df["ex"]*tble1[5,]*100
tble1_re_scaled_1[6,] = sd_df["BurstD2"]/sd_df["ex"]^2*tble1[6,]*100^2
tble1_re_scaled_1[7,] = sd_df["BurstD2"]/sd_df["ex"]^2*tble1[7,]*100^2
tble1_re_scaled_1[8,] = sd_df["BurstD2"]/(sd_df["in."]*sd_df["ex"])*tble1[8,]*100^2

tble1_re_scaled_1

### fit_2

(s = summary(fit_2))

(store = c(s$coefficients[,1],
           sd_df["MeanNrActivationsPerAstrocyte"]
))

write.csv(store,
          file="/home/flo/Projects/Astrocyte_Project/Clean/Data/syn_fits_maa.csv")

# z = matrix(data=NA, nrow = 1e3, ncol = 1e3)
# for (i in c(1:1e3)) {
#   for (j in c(1:1e3)) {
#     z[i,j] = response_2(x[i],y[j],s)
#   }
# }
# 
# z_max = max(z)
# z_min = min(z)
# (z_range = seq(z_min,z_max,length.out = 30))
# (zero_val = length(z_range[z_range<0]))
# 
# col_pal = function(n) {
#   return(c(rep("grey",max(0,times=zero_val-1)),hcl.colors(min(n-zero_val+1,n), "geyser", rev = FALSE)))
# }
# par(mfrow=c(1,1))
# 
# tiff("Plot2.tiff", width = 4, height = 4, units = 'in', res = 300)
# t2 = filled.contour(x,y,z, 
#                     levels = z_range, 
#                     xlab = expression(y[in.]), 
#                     ylab = expression(y[ex]), 
#                     color.palette = col_pal,
#                     cex.axis = 4,
#                     main = "Mean Number of Astrocyte Activations"
# )
# dev.off()


(c = confint(fit_2))
(estimates = t(cbind(attr(s$varcor$`net:con`,"stddev"),
                     s$sigma,
                     t(s$coefficients[,1]))))

(tble2 = cbind("Estimates" = estimates,
               "2.5 CI" = c[,1],
               "97.5 CI" =c[,2]
))
X$in.
tble0_re_scaled_2 = tble2
tble0_re_scaled_2[1,] = sd_df["MeanNrActivationsPerAstrocyte"]*tble2[1,]
tble0_re_scaled_2[2,] = sd_df["MeanNrActivationsPerAstrocyte"]*tble2[2,]
tble0_re_scaled_2[3,] = sd_df["MeanNrActivationsPerAstrocyte"]*tble2[3,]
tble0_re_scaled_2[4,] = sd_df["MeanNrActivationsPerAstrocyte"]/sd_df["in."]*tble2[4,]*100
tble0_re_scaled_2[5,] = sd_df["MeanNrActivationsPerAstrocyte"]/sd_df["ex"]*tble2[5,]*100
tble0_re_scaled_2[6,] = sd_df["MeanNrActivationsPerAstrocyte"]/sd_df["ex"]^2*tble2[6,]*100^2
tble0_re_scaled_2[7,] = sd_df["MeanNrActivationsPerAstrocyte"]/sd_df["ex"]^2*tble2[7,]*100^2
tble0_re_scaled_2[8,] = sd_df["MeanNrActivationsPerAstrocyte"]/(sd_df["in."]*sd_df["ex"])*tble2[8,]*100^2

tble0_re_scaled_2

## Output tables

options(scipen=0) 
# (table_full = cbind(tble0,tble1,tble2))
(table_full = cbind(tble0_re_scaled_0,tble1_re_scaled_1,tble0_re_scaled_2))

(num_formatted_table = formatC(table_full, digits=2, format = "f", flag = "0"))

stargazer(num_formatted_table)

### graphical representation
ranef(fit_0) # random effects for each graph individually
# google --> automatically display estimated curves

# Reproducibility:

Sys.info()
# Originally generated with:

# sysname                                              release 
# "Linux"                                   "5.8.0-43-generic" 
# version                                             nodename 
# "#49~20.04.1-Ubuntu SMP Fri Feb 5 09:57:56 UTC 2021"  "elly" 
# machine                                                login 
# "x86_64"                                                "flo" 
# user                                       effective_user 
# "flo"                                                "flo"