##### Analyse experiments #####
##### Set up files and random seed ####

# set.seed(450)
set.seed(451)
# set.seed(452)

## Import csv
(X_orig = read.csv("/home/flo/Projects/Astrocyte_Project/Clean/Data/allresults_yastro_experiments.csv"))

## Only keep columns of interest
(X = X_orig[,2:length((X_orig[1,]))])

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
fit_0 = lmer(log(BurstD0) ~ y_astro + I(y_astro^2) + (y_astro || net:con), data=X)
fit_1 = lmer(log(BurstD2) ~ y_astro + I(y_astro^2) + (y_astro || net:con), data=X)
fit_2 = lmer(log(MeanNrActivationsPerAstrocyte) ~ y_astro + I(y_astro^2) + (y_astro || net:con), data=X)


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

# summaries ####

response = function(x,s){
  return(exp(s$coefficients[1]
            + s$coefficients[2]*x
            + s$coefficients[3]*x^2))
}

(x = seq(min(X["y_astro"]),max(X["y_astro"]), length.out = 1e3))

(store = c(min(X["y_astro"]),
           max(X["y_astro"])
))

write.csv(store,
          file="/home/flo/Projects/Astrocyte_Project/Clean/Data/yastro_plot_boundaries.csv")

## fit 0

(s = summary(fit_0))

(store = c(s$coefficients[1],
           s$coefficients[2],
           s$coefficients[3],
           attr(s$varcor, "sc"),
           attr(s$varcor$net.con, "stddev")[1],
           attr(s$varcor$net.con.1, "stddev")
))

write.csv(store,
          file="/home/flo/Projects/Astrocyte_Project/Clean/Data/yastro_fits_msr.csv")

(c = confint(fit_0))

(estimates = t(cbind(attr(s$varcor$net.con, "stddev"),
                     attr(s$varcor$net.con.1, "stddev")[1],
                     attr(s$varcor, "sc"),
                     t(s$coefficients[,1]))))

(tble0 = cbind("Estimates" = estimates,
               "2.5 CI" = c[,1],
               "97.5 CI"= c[,2]
))

## fit 1

(s = summary(fit_1))

(store = c(s$coefficients[1],
           s$coefficients[2],
           s$coefficients[3],
           attr(s$varcor, "sc"),
           attr(s$varcor$net.con, "stddev")[1],
           attr(s$varcor$net.con.1, "stddev")
           ))

write.csv(store,
          file="/home/flo/Projects/Astrocyte_Project/Clean/Data/yastro_fits_mbr.csv",
          sep=",")

(c = confint(fit_1))

(estimates = t(cbind(attr(s$varcor$net.con, "stddev"),
                     attr(s$varcor$net.con.1, "stddev")[1],
                     attr(s$varcor, "sc"),
                     t(s$coefficients[,1]))))

(tble1 = cbind("Estimates" = estimates,
               "2.5 CI" = c[,1],
               "97.5 CI"= c[,2]
))

## fit 2

(s = summary(fit_2))

attr(s$varcor$net.con, "stddev")

(store = c(s$coefficients[1],
           s$coefficients[2],
           s$coefficients[3],
           attr(s$varcor, "sc"),
           attr(s$varcor$net.con, "stddev")[1],
           attr(s$varcor$net.con.1, "stddev")
))

write.csv(store,
          file="/home/flo/Projects/Astrocyte_Project/Clean/Data/yastro_fits_maa.csv")

(c = confint(fit_2))

(estimates = t(cbind(attr(s$varcor$net.con, "stddev"),
                     attr(s$varcor$net.con.1, "stddev")[1],
                     attr(s$varcor, "sc"),
                     t(s$coefficients[,1]))))

(tble2 = cbind("Estimates" = estimates,
               "2.5 CI" = c[,1],
               "97.5 CI"= c[,2]
))

## Output tables

options(scipen=0) 
# (table_full = cbind(tble0,tble1,tble2))
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