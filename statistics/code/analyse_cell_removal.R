##### Analyse experiments #####
##### Set up files, preprocessing and random seed ####

# set.seed(450)
# set.seed(451)
# set.seed(452)
# set.seed(453)
# set.seed(454)
# set.seed(455)
# set.seed(456)
set.seed(457)

## Import csv
# (X_orig = read.csv("/home/flo/Projects/Astrocyte_Project/reduced_networks_updated.csv"))
(X_orig = read.csv("/home/flo/Projects/Astrocyte_Project/allresults_reduction_experiments.csv"))

## Only keep columns of interest
(X = X_orig[,2:length((X_orig[1,]))])

## Look at the structure of the data
str(X)

## Re-configure as categorical
X$net = as.factor(X$net)
X$con = as.factor(X$con)
X$reductionType = as.factor(X$reductionType)
X$variation = as.factor(X$variation)

# Structure of data after reconfiguration
# reductionType == 1 :> reduced astrocytes
# reductionType == 2 :> reduced neurons

X$AstroDrop[X["reductionType"]==0] = 0
X$AstroDrop[(X["reductionType"]==1)&(X["variation"]==0)] = 1
X$AstroDrop[(X["reductionType"]==1)&(X["variation"]==1)] = 2
X$AstroDrop[(X["reductionType"]==1)&(X["variation"]==2)] = 3
X$AstroDrop[(X["reductionType"]==2)] = NA

X$NeuroDrop[X["reductionType"]==0] = 0
X$NeuroDrop[(X["reductionType"]==2)&(X["variation"]==0)] = 1
X$NeuroDrop[(X["reductionType"]==2)&(X["variation"]==1)] = 2
X$NeuroDrop[(X["reductionType"]==2)&(X["variation"]==2)] = 3
X$NeuroDrop[(X["reductionType"]==1)] = NA

X$AstroDrop = as.factor(X$AstroDrop)
X$NeuroDrop = as.factor(X$NeuroDrop)

str(X)

(Astro_subset = na.omit(subset(X, select = -NeuroDrop)))
(Neuro_subset = na.omit(subset(X, select = -AstroDrop)))

##### Plots and exploratory analysis #####
par(mfrow = c(2,3))
boxplot(log(BurstD0) ~ AstroDrop, data=Astro_subset)
boxplot(log(BurstD2) ~ AstroDrop, data=Astro_subset)
boxplot(log(meanAA) ~ AstroDrop, data=Astro_subset)

boxplot(log(BurstD0) ~ NeuroDrop, data=Neuro_subset)
boxplot(log(BurstD2) ~ NeuroDrop, data=Neuro_subset)
boxplot((meanAA)~ NeuroDrop, data=Neuro_subset)
boxplot((meanAA)^{0.5} ~ NeuroDrop, data=Neuro_subset)


##### Analysis using mixed effects model ####

# Required packages
library(lmerTest)
library(multcomp)
library(stargazer)

# set contrasts for result interpretation
options(contrasts = c("contr.treatment", "contr.poly"))
# options(contrasts = c("contr.sum", "contr.poly"))

## mixed effects model: quantify random effect of different drops ####
fit_A_0 <- lmer(log(BurstD0) ~ AstroDrop + (1|net:con:AstroDrop), data = Astro_subset)
fit_A_1 <- lmer(log(BurstD2) ~ AstroDrop + (1|net:con:AstroDrop), data = Astro_subset)
fit_A_2 <- lmer(log(MeanNrActivationsPerAstrocyte) ~ AstroDrop + (1|net:con:AstroDrop), data = Astro_subset)
# fit_A_2 <- lmer((meanAA)^{0.5} ~ AstroDrop + (1|net:con:AstroDrop), data = Astro_subset)


fit_N_0 <- lmer(log(BurstD0) ~ NeuroDrop + (1|net:con:NeuroDrop), data = Neuro_subset)
fit_N_1 <- lmer(log(BurstD2) ~ NeuroDrop + (1|net:con:NeuroDrop), data = Neuro_subset)
fit_N_2 <- lmer((MeanNrActivationsPerAstrocyte) ~ NeuroDrop + (1|net:con:NeuroDrop), data = Neuro_subset)
# fit_N_2 <- lmer(log(MeanNrActivationsPerAstrocyte+1e-5) ~ NeuroDrop + (1|net:con:NeuroDrop), data = Neuro_subset)
fit_N_2 <- lmer((MeanNrActivationsPerAstrocyte) ~ NeuroDrop + (1|net:con:NeuroDrop), data = Neuro_subset)
# fit_N_2 <- lmer((MeanNrActivationsPerAstrocyte)^{0.5} ~ NeuroDrop + (1|net:con:NeuroDrop), data = Neuro_subset)
# fit_N_2 <- lmer(log(meanAA+1e-5) ~ NeuroDrop + (1|net:con:NeuroDrop), data = Neuro_subset)
# fit_N_2 <- lmer((meanAA)^{0.5} ~ NeuroDrop + (1|net:con:NeuroDrop), data = Neuro_subset)


## using contrasts test fixed effect of dropping cells
fit_cont_A_0 = glht(fit_A_0, linfct = mcp(AstroDrop = c(-1,+1/3,+1/3,+1/3)))
fit_cont_A_1 = glht(fit_A_1, linfct = mcp(AstroDrop = c(-1,+1/3,+1/3,+1/3)))
fit_cont_A_2 = glht(fit_A_2, linfct = mcp(AstroDrop = c(-1,+1/3,+1/3,+1/3)))

fit_cont_N_0 = glht(fit_N_0, linfct = mcp(NeuroDrop = c(-1,+1/3,+1/3,+1/3)))
fit_cont_N_1 = glht(fit_N_1, linfct = mcp(NeuroDrop = c(-1,+1/3,+1/3,+1/3)))
fit_cont_N_2 = glht(fit_N_2, linfct = mcp(NeuroDrop = c(-1,+1/3,+1/3,+1/3)))

summary(fit_cont_N_2)


exp(-1/3*(1.05+1.10+1.09))

# TA and qq-plots ####
plot(fit_A_0)
par(mfrow=c(2,1))
qqnorm(ranef(fit_A_0)$'net:con:AstroDrop'[,1], main = "Connectivity QQ-Plot")
qqline(ranef(fit_A_0)$'net:con:AstroDrop'[,1], main = "Connectivity QQ-Plot")
qqnorm(resid(fit_A_0), main = "Residuals QQ-Plot")
qqline(resid(fit_A_0), main = "Residuals QQ-Plot")

plot(fit_A_1)
par(mfrow=c(2,1))
qqnorm(ranef(fit_A_1)$'net:con:AstroDrop'[,1], main = "Connectivity QQ-Plot")
qqline(ranef(fit_A_1)$'net:con:AstroDrop'[,1], main = "Connectivity QQ-Plot")
qqnorm(resid(fit_A_1), main = "Residuals QQ-Plot")
qqline(resid(fit_A_1), main = "Residuals QQ-Plot")

plot(fit_A_2)
par(mfrow=c(2,1))
qqnorm(ranef(fit_A_2)$'net:con:AstroDrop'[,1], main = "Connectivity QQ-Plot")
qqline(ranef(fit_A_2)$'net:con:AstroDrop'[,1], main = "Connectivity QQ-Plot")
qqnorm(resid(fit_A_2), main = "Residuals QQ-Plot")
qqline(resid(fit_A_2), main = "Residuals QQ-Plot")

plot(fit_N_0)
par(mfrow=c(2,1))
qqnorm(ranef(fit_N_0)$'net:con:NeuroDrop'[,1], main = "Connectivity QQ-Plot")
qqline(ranef(fit_N_0)$'net:con:NeuroDrop'[,1], main = "Connectivity QQ-Plot")
qqnorm(resid(fit_N_0), main = "Residuals QQ-Plot")
qqline(resid(fit_N_0), main = "Residuals QQ-Plot")

plot(fit_N_1)
par(mfrow=c(2,1))
qqnorm(ranef(fit_N_1)$'net:con:NeuroDrop'[,1], main = "Connectivity QQ-Plot")
qqline(ranef(fit_N_1)$'net:con:NeuroDrop'[,1], main = "Connectivity QQ-Plot")
qqnorm(resid(fit_N_1), main = "Residuals QQ-Plot")
qqline(resid(fit_N_1), main = "Residuals QQ-Plot")

plot(fit_N_2)
par(mfrow=c(2,1))
qqnorm(ranef(fit_N_2)$'net:con:NeuroDrop'[,1], main = "Connectivity QQ-Plot")
qqline(ranef(fit_N_2)$'net:con:NeuroDrop'[,1], main = "Connectivity QQ-Plot")
qqnorm(resid(fit_N_2), main = "Residuals QQ-Plot")
qqline(resid(fit_N_2), main = "Residuals QQ-Plot")

# summaries and confints ####

## A_0
(s = summary(fit_A_0))
r = confint(fit_A_0)
(c = confint(fit_cont_A_0))

(res = t(cbind(attr(s$varcor$`net:con:AstroDrop`,"stddev"),
               s$sigma,
               t(s$coefficients[,1]))))
(interest_table0 = rbind(cbind("Estimates"=res,
                               "lower" = r[,1],
                               "upper" = r[,2]),
                         c(c$confint[1,1],
                           c$confint[1,3],
                           c$confint[1,2])))

## A_1
s = summary(fit_A_1)
r = confint(fit_A_1)
c = confint(fit_cont_A_1)

(res = t(cbind(attr(s$varcor$`net:con:AstroDrop`,"stddev"),
               s$sigma,
               t(s$coefficients[,1]))))
(interest_table1 = rbind(cbind("Estimates"=res,
                               "lower" = r[,1],
                               "upper" = r[,2]),
                         c(c$confint[1,1],
                           c$confint[1,3],
                           c$confint[1,2])))

## A_2
s = summary(fit_A_2)
r = confint(fit_A_2)
c = confint(fit_cont_A_2)

(res = t(cbind(attr(s$varcor$`net:con:AstroDrop`,"stddev"),
               s$sigma,
               t(s$coefficients[,1]))))
(interest_table2 = rbind(cbind("Estimates"=res,
                               "lower" = r[,1],
                               "upper" = r[,2]),
                         c(c$confint[1,1],
                           c$confint[1,3],
                           c$confint[1,2])))

## N_0
s = summary(fit_N_0)
r = confint(fit_N_0)
c = confint(fit_cont_N_0)

(res = t(cbind(attr(s$varcor$`net:con:NeuroDrop`,"stddev"),
               s$sigma,
               t(s$coefficients[,1]))))
(interest_table3 = rbind(cbind("Estimates"=res,
                               "lower" = r[,1],
                               "upper" = r[,2]),
                         c(c$confint[1,1],
                           c$confint[1,3],
                           c$confint[1,2])))

## N_1
s = summary(fit_N_1)
r = confint(fit_N_1)
(c = confint(fit_cont_N_1))

(res = t(cbind(attr(s$varcor$`net:con:NeuroDrop`,"stddev"),
               s$sigma,
               t(s$coefficients[,1]))))
(interest_table4 = rbind(cbind("Estimates"=res,
                               "lower" = r[,1],
                               "upper" = r[,2]),
                         c(c$confint[1,1],
                           c$confint[1,3],
                           c$confint[1,2])))

## N_2
(s = summary(fit_N_2))
r = confint(fit_N_2)
(r_bar = confint(fit_N_2,level=0.975))
#(r_bar = confint(fit_N_2,level=0.1))
(c = confint(fit_cont_N_2))
(c_bar = confint(fit_cont_N_2,level=0.975))
# (c_bar = confint(fit_cont_N_2,level=0.5))

s$sigma
s$varcor

s

(res = t(cbind(attr(s$varcor$`net:con:NeuroDrop`,"stddev"),
               s$sigma,
              t(s$coefficients[,1]))))
(interest_table5 = rbind(cbind("Estimates"=res,
                               "lower" = r[,1],
                               "upper" = r[,2]),
                         c(c$confint[1,1],
                           c$confint[1,3],
                           c$confint[1,2])))
(lower_mu = r_bar[3,1])
(upper_mu = r_bar[3,2])
(lower_Delta = -c$confint[1,3]) 
(upper_Delta= -c$confint[1,2])

## since the reduction can be at most 1, the CI is cut off:
(special_CIs = c(max(0,lower_Delta/upper_mu),-c$confint[1,1]/res[3],min(upper_Delta/lower_mu,1)))

## Summarised Table

(astro_full_mat = cbind(interest_table0,interest_table1,interest_table2))
(neuro_full_mat = cbind(interest_table3,interest_table4,interest_table5))

#(full_astro = round(astro_full_mat,2))
#(full_neuro = round(neuro_full_mat,2))

# c(astro_full_mat[1,1:3]/astro_full_mat[3,1],astro_full_mat[1,4:6]/astro_full_mat[3,4],astro_full_mat[1,7:9]/astro_full_mat[3,7])

(main_astro = (rbind(1-exp(astro_full_mat[7,]),
                     c(astro_full_mat[1,1:3]/astro_full_mat[3,1],astro_full_mat[1,4:6]/astro_full_mat[3,4],astro_full_mat[1,7:9]/astro_full_mat[3,7]),
                     c(astro_full_mat[2,1:3]/astro_full_mat[3,1],astro_full_mat[2,4:6]/astro_full_mat[3,4],astro_full_mat[2,7:9]/astro_full_mat[3,7])
                     )))
(main_neuro = (rbind(c(1-exp(neuro_full_mat[7,1:6]),special_CIs),
                     c(neuro_full_mat[1,1:3]/neuro_full_mat[3,1],neuro_full_mat[1,4:6]/neuro_full_mat[3,4],neuro_full_mat[1,7:9]/neuro_full_mat[3,7]),
                     c(neuro_full_mat[2,1:3]/neuro_full_mat[3,1],neuro_full_mat[2,4:6]/neuro_full_mat[3,4],neuro_full_mat[2,7:9]/neuro_full_mat[3,7])
                    )))

(num_formatted_table_astro = formatC(astro_full_mat, digits=2, format = "f", flag = "0"))
stargazer(num_formatted_table_astro)
(num_formatted_table_neuro = formatC(neuro_full_mat, digits=2, format = "f", flag = "0"))
stargazer(num_formatted_table_neuro)

(num_formatted_main_astro = formatC(main_astro*100, digits=2, format = "f", flag = "0"))
stargazer(num_formatted_main_astro)
(num_formatted_main_neuro = formatC(main_neuro*100, digits=2, format = "f", flag = "0"))
stargazer(num_formatted_main_neuro)
## (main_neuro = round(rbind((neuro_full_mat[6,]),neuro_full_mat[1,],neuro_full_mat[2,]),2)). Need to do something else here, since we used the sqrt transform.



# (full1 = (cbind(paste(full[,1],full[,2],sep="ç"),paste(full[,7],full[,8],sep="ç"),
#                 paste(full[,3],full[,4],sep="ç"),paste(full[,9],full[,10],sep="ç"),
#                paste(full[,5],full[,6],sep="ç"),paste(full[,11],full[,12],sep="ç")
#                )))
# row.names(full1) = c("1","2","3","4","5","6")
# colnames(full1) = c("c","c","c","c","c","c")
# full1


# OLD CONFIDENCE INTERVALS AND SUMMARIES ####

summary(fit_A_1)
confint(fit_A_1)
stargazer(confint(fit_A_1))
stargazer(summary(fit_A_1)$coefficients)

summary(fit_A_2)
confint(fit_A_2)
stargazer(confint(fit_A_2))
stargazer(summary(fit_A_2)$coefficients)

summary(fit_N_0)
confint(fit_N_0)
stargazer(confint(fit_N_0))
stargazer(summary(fit_N_0)$coefficients)

summary(fit_N_1)
confint(fit_N_1)
stargazer(confint(fit_N_1))
stargazer(summary(fit_N_1)$coefficients)

summary(fit_N_2)
confint(fit_N_2)
stargazer(confint(fit_N_2))
stargazer(summary(fit_N_2)$coefficients)


# summaries and confints
summary(fit_cont_A_0)
confint(fit_cont_A_0)
stargazer(confint(fit_cont_A_0))
stargazer(summary(fit_cont_A_0)$coefficients)

summary(fit_cont_A_1)
confint(fit_cont_A_1)
stargazer(confint(fit_cont_A_1))
stargazer(summary(fit_cont_A_1)$coefficients)

summary(fit_cont_A_2)
confint(fit_cont_A_2)
stargazer(confint(fit_cont_A_2))
stargazer(summary(fit_cont_A_2)$coefficients)

summary(fit_cont_N_0)
confint(fit_cont_N_0)
stargazer(confint(fit_cont_N_0))
stargazer(summary(fit_cont_N_0)$coefficients)

summary(fit_cont_N_1)
confint(fit_cont_N_1)
stargazer(confint(fit_cont_N_1))
stargazer(summary(fit_cont_N_1)$coefficients)

summary(fit_cont_N_2)
confint(fit_cont_N_2)
stargazer(confint(fit_cont_N_2))
stargazer(summary(fit_cont_N_2)$coefficients)

##### System info ####
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