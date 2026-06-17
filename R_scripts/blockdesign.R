#Trial_design_incubator_script_2023-02-13
library(blocksdesign)
library(dplyr)
.libPaths("/Users/pro/Desktop/")
setwd("/Users/pro/Desktop/")

# 1080 total dishes
# 2 groups in each line: for jars or exp 
# 108 dishes per shelf
#Each tray has two layers
#bottom layer = 8 x 9 = 72
#top layer = 108 - 72 = 36
# 26 magic lines + Col0 with 40 reps each
#layout treatments
treatments<-factor(c(rep(1:27,each=40)))
#form first level of blocking 
group=factor(rep(1:2,each=540))
#form second level of blocking 
shelf=factor(rep(1:10,each=108))
#form third level of blocking 
unequal_pat <- rep(c(72, 36), times=10)
layer=factor(rep(1:20,unequal_pat))
#combine the two
blocking=data.frame(group,shelf,layer)
#check block object
table(blocking$group)
table(blocking$shelf)
table(blocking$layer)
#check treatments
table(treatments)
nrow(blocking)
length(treatments)==nrow(blocking)
#run function 
date()
nem.des=design(treatments, blocking, searches = 1000, jumps = 5)
date()
table(table(nem.des$Design$treatments))
table(as.list(aggregate(treatments~group,data=nem.des$Design,table))[[2]])
table(as.list(aggregate(treatments~shelf,data=nem.des$Design,table))[[2]])
nem.des$Blocks_model
write.csv(nem.des$Design,"Magic_Screen_nematodes_Trial_design_2022_s1000_j5.csv",row.names = F)
capture.output(nem.des, file = "Magic_Screen_nematodes_Trialdesign_object_2022.csv")
writeLines(capture.output(sessionInfo()), "sessionInfo.txt")

