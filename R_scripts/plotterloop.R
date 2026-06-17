library(CMplot)
library(hrbrthemes)
library(tidyfst)
source("https://raw.githubusercontent.com/YinLiLin/CMplot/master/R/CMplot.r")

# Set the working directory
setwd("/Users/pro/Desktop")

# Get a list of all files ending with .logP.txt
files <- list.files(pattern = "\\.logP\\.txt$")

# Loop through each file
for (file in files) {
  # Read the file
  data <- read.table(file, header = TRUE)
  
  # Convert the data to a data frame
  DT <- data.frame(data)
  
  # Extract the filename without the extension for the column name
  file_base <- sub("\\.logP\\.txt$", "", file)
  
  # Rename the columns dynamically based on the filename
  colnames(DT) <- c("chr", "position", file_base)
  
  # Add a new column for SNP (rownames)
  DT$SNP <- rownames(DT)
  
  # Reorder the columns
  DT <- DT[, c(4, 1, 2, 3)]
  
  # Print the first few rows to check
  head(DT)
  
  # Plot using CMplot
  CMplot(DT, plot.type = "m",
         col = c("#ac92eb", "#4fc1e8", "#a0d568", "#ffce54", "#ed5564"),
         LOG10 = FALSE, ylim = c(0, 8),  threshold=c(6),
         threshold.lwd = c(1, 1), threshold.col = c("black", "grey"), 
         amplify = TRUE, bin.size = 1e6, multracks = FALSE, 
         points.alpha = 10L, cex = c(0.2, 0.2, 0.2),
         chr.den.col = c("darkgreen", "yellow", "red"),
         signal.col = c("black"), signal.cex = c(0.3, 0.3),
         signal.pch = c(19, 19), file = "jpg", dpi = 300, 
         file.output = TRUE, verbose = TRUE,
         width = 14, height = 6)
}

