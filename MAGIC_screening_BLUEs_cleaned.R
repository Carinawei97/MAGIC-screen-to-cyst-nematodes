
setwd("/Users/pro/Desktop/Distribution.v1")
library(lme4); library(lmerTest); library(pbkrtest)
library(data.table)
library(tidyverse)
library(emmeans)
library(tidyfst)
library(dplyr)
#static_41dpi.csv

data = fread ("merged.csv",data.table = F)
head(data)
str(data)


#remove lines with replicates less than 5
lines_rep = count(data, lines)
str(lines_rep)
set.seed(1L)
Df <- as.data.frame.array(lines_rep)
DT <- as.data.frame.array(data)
class(Df)
class(DT)
r_lines <- filter_dt(Df,n %between% c(0,4))
print(r_lines$lines)
data2 <- data[! data$lines %in% r_lines$lines, ]
head(data2)
str(data2)

#setting the factors
col =1:12
col2=13:20
data2[,col] = data2 %>% select(all_of(col)) %>% map_df(as.factor)
data2[,col2] = data2 %>% select(all_of(col2)) %>% map_df(as.numeric)
str(data2)


mm1 <- lmer( y_for_steepest_slope  ~ lines +(1|lines:reps)  + (1|bench) + (1|bench:tray) + (1|bench:tray:sorted_Elayer) 
             + (1|innoculator)+ (1|innoculator:jars)+ (1|innoculator:jars:harvest_time)
             ,REML=T,na.action = na.exclude, data=data2)

step(mm1)
summary(mm1)
#no matter what the stepwise tell you the best model is, you have to include 'reps' as a random factor
newmm1<- lmer( y_for_steepest_slope ~ lines + (1|lines:reps)+ (1 | innoculator),
              na.action = na.exclude, data=data2)
summary(newmm1)


options(max.print = 15000)
emm_options(rg.limit = 15000)
emm_options(pbkrtest.limit = 15000)
emm_options(lmerTest.limit = 15000)
BLUEs<-emmeans(newmm1, "lines","reps","id")
BLUEs <-as.data.frame(BLUEs)
write.csv(BLUEs, "number41dpi_BLUEs.csv") 

df4 = fread ("clean2_dynamic_gwas.csv",data.table = F)
write.table(df4,"fit_dynamic_gwas.txt",row.names = F,sep ="\t", quote = F, col.names = T)

Flibrary(lsmeans)
lsmeans <- lsmeans::lsmeans
lsmeans(mm1, pairwise ~ lines)
lsmeans(mm1, pairwise ~ lines, adjust="none")


