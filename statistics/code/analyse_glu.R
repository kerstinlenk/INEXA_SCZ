##### Analyse experiments #####
##### Set up files and random seed ####

# set.seed(450)
set.seed(451)
# set.seed(452)
# set.seed(453)
# set.seed(454)
# set.seed(455)
# set.seed(456)
# set.seed(457)

## Import csv

# (X_orig1 = read.csv("/home/flo/Projects/Astrocyte_Project/res-with-its-210217.csv"))
# (X_orig2 = read.csv("/home/flo/Projects/Astrocyte_Project/all_in_ex.csv"))
(X_orig = read.csv("/home/flo/Projects/Astrocyte_Project/Clean/Data/allresults_glut_changes.csv"))

# need to rename columns: param_F and param_G were accidentally swapped, so they need to be swapped back:
names(X_orig)[names(X_orig) == "param_F"] <- "tmp"
names(X_orig)[names(X_orig) == "param_G"] <- "param_F"
names(X_orig)[names(X_orig) == "tmp"] <- "param_G"

## Only keep columns of interest
(X = X_orig[,2:length((X_orig[1,]))])

## Look at the structure of the data
str(X)

## Re-configure as categorical
X$param_G = 1/X$param_G
X$param_F = 1/X$param_F

X$net = as.factor(X$net)
X$con = as.factor(X$con)

# Structure of data after reconfiguration
str(X)
(data = X)

##### Interaction and exploratory analysis ##### 
par(mfrow=c(3,1))

## Interaction of Omega_G with Omega_F: 
# Does the slope when going between values of omega_G depend on Omega_F?
with(data, interaction.plot(x.factor = interaction(in,ex),
                            trace.factor = interaction(param_G,param_F),
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
boxplot(BurstD0~interaction(param_G,param_F), data=data)
boxplot(BurstD2~interaction(param_G,param_F), data=data)
boxplot(MeanAA~interaction(param_G,param_F), data=data)

par(mfrow=c(3,1))
boxplot(BurstD0~interaction(param_F,param_G), data=data)
boxplot(BurstD2~interaction(param_F,param_G), data=data)
boxplot(MeanAA~interaction(param_F,param_G), data=data)

##### Analysis using mixed effects model ####

# Required packages
library(lmerTest)
library(multcomp)
library(stargazer)

options(contrasts = c("contr.treatment", "contr.poly"))

# fit models
# two options, random intersept and random slope (1st case), only random intersept (2nd case):
fit_0 = lmer(BurstD0 ~ param_F + param_G + I(param_F^2) + I(param_G^2) + param_F:param_G + (param_F + param_G + param_F:param_G || net:con), data=X)
fit_1 = lmer(BurstD2 ~ param_F + param_G + I(param_F^2) + I(param_G^2) + param_F:param_G + (param_F + param_G + param_F:param_G || net:con), data=X)
fit_2 = lmer(MeanNrActivationsPerAstrocyte ~ param_F + param_G + I(param_F^2) + I(param_G^2) + param_F:param_G + (param_F + param_G + param_F:param_G || net:con), data=X)
# fit_2 = lmer(MeanAA ~ param_F + param_G + (param_F + param_G || net:con), data=X)

# diagnostic TA plots, qq-plots ####
par(mfrow=c(1,1))
plot(fit_0)
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

# summaries ####
### x = param_F, y = param_G at the moment.
response = function(x,y,s){
  return(s$coefficients[1]+s$coefficients[2]*x+s$coefficients[3]*y+s$coefficients[6]*x*y+s$coefficients[4]*x^2+s$coefficients[5]*y^2)
}

(s = summary(fit_0))

(store = c(s$coefficients[,1]))

write.csv(store,
          file="/home/flo/Projects/Astrocyte_Project/Clean/Data/glu_fits_msr.csv")

# (x = seq(0.5,5, length.out = 1e3))
# (y = seq(0.5,5, length.out = 1e3))

(x = seq(min(X$param_F),max(X$param_F), length.out = 1e3))
(y = seq(min(X$param_G),max(X$param_G), length.out = 1e3))

(store = c(min(X$param_F),
           max(X$param_F),
           min(X$param_G),
           max(X$param_G)
))

write.csv(store,
          file="/home/flo/Projects/Astrocyte_Project/Clean/Data/glu_plot_boundaries.csv")


# z = matrix(data=NA, nrow = 1e3, ncol = 1e3)
# for (i in c(1:1e3)) {
#   # print(i)
#   for (j in c(1:1e3)) {
#     z[i,j] = response(x[i],y[j],s)
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
# 
# t0 = filled.contour(x,y,z, 
#                     levels = z_range, 
#                     xlab = expression(w[f]), 
#                     ylab = expression(w[g]), 
#                     color.palette = col_pal,
#                     main = "Mean Spike Rate"
# )


(c = confint(fit_0))
(estimates = t(cbind(attr(s$varcor$`net.con`,"stddev"),
      attr(s$varcor$`net.con.1`,"stddev"),
      attr(s$varcor$`net.con.2`,"stddev"),
      attr(s$varcor$`net.con.3`,"stddev"),
      s$sigma,
      t(s$coefficients[,1]))))

(tble0 = cbind("Estimates" = estimates,
              #"midpoints" = 1/2*((c[1:5,2])+(c[1:5,1])),
              "2.5 CI" = c[,1],
              "97.5 CI" =c[,2]
              #"diffs" = 1/2*((c[1:5,2])-(c[1:5,1])),
              #"upper-est" = c[1:5,2]-estimates,
              #"est-lower" = estimates-(c[1:5,1])
              ))

### fit1

(s = summary(fit_1))

(store = c(s$coefficients[,1]))

write.csv(store,
          file="/home/flo/Projects/Astrocyte_Project/Clean/Data/glu_fits_mbr.csv")

# z = matrix(data=NA, nrow = 1e3, ncol = 1e3)
# for (i in c(1:1e3)) {
#   # print(i)
#   for (j in c(1:1e3)) {
#     z[i,j] = response(x[i],y[j],s)
#   }
# }
# z_max = max(z)
# z_min = min(z)
# (z_range = seq(z_min,z_max,length.out = 30))
# (zero_val = length(z_range[z_range<0]))
# 
# col_pal = function(n) {
#   return(c(rep("grey",max(0,times=zero_val-1)),hcl.colors(min(n-zero_val+1,n), "geyser", rev = FALSE)))
# }
# 
# t1 = filled.contour(x,y,z, 
#                     levels = z_range, 
#                     xlab = expression(w[f]), 
#                     ylab = expression(w[g]), 
#                     color.palette = col_pal,
#                     main = "Mean Burst Rate"
# )

(c = confint(fit_1))
(estimates = t(cbind(attr(s$varcor$`net.con`,"stddev"),
                     attr(s$varcor$`net.con.1`,"stddev"),
                     attr(s$varcor$`net.con.2`,"stddev"),
                     attr(s$varcor$`net.con.3`,"stddev"),
                     s$sigma,
                     t(s$coefficients[,1]))))

(tble1 = cbind("Estimates" = estimates,
               #"midpoints" = 1/2*((c[1:5,2])+(c[1:5,1])),
               "2.5 CI" = c[,1],
               "97.5 CI" =c[,2]
               #"diffs" = 1/2*((c[1:5,2])-(c[1:5,1])),
               #"upper-est" = c[1:5,2]-estimates,
               #"est-lower" = estimates-(c[1:5,1])
))


### fit_2
(s = summary(fit_2))

(store = c(s$coefficients[,1]))

write.csv(store,
          file="/home/flo/Projects/Astrocyte_Project/Clean/Data/glu_fits_maa.csv")

# z = matrix(data=NA, nrow = 1e3, ncol = 1e3)
# for (i in c(1:1e3)) {
#   #print(i)
#   for (j in c(1:1e3)) {
#     z[i,j] = response(x[i],y[j],s)
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
# 
# t2 = filled.contour(x,y,z, 
#                     levels = z_range, 
#                     xlab = expression(w[f]), 
#                     ylab = expression(w[g]), 
#                     color.palette = col_pal,
#                     main = "Mean Number of Astrocyte Activations"
# )

(c = confint(fit_2))
(estimates = t(cbind(attr(s$varcor$`net.con`,"stddev"),
                     attr(s$varcor$`net.con.1`,"stddev"),
                     attr(s$varcor$`net.con.2`,"stddev"),
                     attr(s$varcor$`net.con.3`,"stddev"),
                     s$sigma,
                     t(s$coefficients[,1]))))

(tble2 = cbind("Estimates" = estimates,
               #"midpoints" = 1/2*((c[1:5,2])+(c[1:5,1])),
               "2.5 CI" = c[,1],
               "97.5 CI" =c[,2]
               #"diffs" = 1/2*((c[1:5,2])-(c[1:5,1])),
               #"upper-est" = c[1:5,2]-estimates,
               #"est-lower" = estimates-(c[1:5,1])
))

options(scipen=0) 
(table_full = cbind(tble0,tble1,tble2))

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