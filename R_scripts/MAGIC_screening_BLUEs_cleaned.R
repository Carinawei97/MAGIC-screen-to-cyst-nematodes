
setwd("/Users/pro/Desktop/Distribution.v1") #setting the director
library(lme4); library(lmerTest); library(pbkrtest)
library(data.table)
library(tidyverse)
library(emmeans)
library(tidyfst)
library(dplyr)
#static_41dpi.csv

data = fread ("example_1.csv",data.table = F) # an example of the file you need for the BLUEs
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

#setting the factors, make sure all the enviornmental factors are setting as factor and the phenotype as numeric
col =1:12
col2=13:20
data2[,col] = data2 %>% select(all_of(col)) %>% map_df(as.factor)
data2[,col2] = data2 %>% select(all_of(col2)) %>% map_df(as.numeric)
str(data2)
#standard model for all the phenotypes, you need to start with this
#replace the phenotype as the real phenotype yuo are working on, the data is the csv file you just defined and cleaned
mm1 <- lmer( phenotype  ~ lines +(1|lines:reps)  + (1|bench) + (1|bench:tray) + (1|bench:tray:sorted_Elayer) 
             + (1|innoculator)+ (1|innoculator:jars)+ (1|innoculator:jars:harvest_time)
             ,REML=T,na.action = na.exclude, data=data2)

step(mm1) #stepwise regression to find the best linear model for your phenotype, can be different each time
summary(mm1) #unnessary step unless your best model is the same as the standard model
#no matter what the stepwise tell you the best model is, you have to include 'reps' as a random factor
#this is the BLUEs we need with the best model, copy paste the model after the step(mm1), replace the model below
newmm1<- lmer( y_for_steepest_slope ~ lines + (1|lines:reps)+ (1 | innoculator),
              na.action = na.exclude, data=data2)
#get the blues result from the best model
summary(newmm1)

#unlimit the print limit according to your model size, accroding to how many individual estimators you have
options(max.print = 15000)
emm_options(rg.limit = 15000)
emm_options(pbkrtest.limit = 15000)
emm_options(lmerTest.limit = 15000)
# put the BLUEs result to csv dataframe
BLUEs<-emmeans(newmm1, "lines","reps","id") # define the dataframe you would like, you can decide to only keep the lines level
BLUEs <-as.data.frame(BLUEs)
#here is the result after the BLUEs
write.csv(BLUEs, "number41dpi_BLUEs.csv") 
# in case yuo need the txt for GWAS
df4 = fread ("clean2_dynamic_gwas.csv",data.table = F)
write.table(df4,"fit_dynamic_gwas.txt",row.names = F,sep ="\t", quote = F, col.names = T)

#pairwise study after the BLUEs, no need to use the csv file but directly from the model matrix,  but only working at line level
Flibrary(lsmeans)
lsmeans <- lsmeans::lsmeans
lsmeans(mm1, pairwise ~ lines)
lsmeans(mm1, pairwise ~ lines, adjust="none")


