# Required libraries
library(ggplot2)
library(ggpmisc)  # For equation annotations
library(openxlsx)  # For Excel export

# Load your data
data <- read.table("/Users/pro/Desktop/Dynamic/combined_traits.txt", header = TRUE, sep = "\t")

# Remove the first column (subject names)
phenotypes <- data[, -1]

# Calculate the correlation matrix
cor_matrix <- cor(phenotypes, use = "pairwise.complete.obs")

# Export correlation matrix to Excel
write.csv(cor_matrix, file = "correlation_matrix_2.csv")

# Create a folder to save the plots
dir.create("plots")

# Generate scatter plots for each pair of phenotypes
for(i in 1:(ncol(phenotypes)-1)){
  for(j in (i+1):ncol(phenotypes)){
    p <- ggplot(phenotypes, aes_string(x = names(phenotypes)[i], y = names(phenotypes)[j])) +
      geom_point() +
      geom_smooth(method = "lm", formula = y ~ x, se = FALSE) +
      stat_poly_eq(formula = y ~ x, 
                   aes(label = paste(..eq.label.., ..rr.label.., ..p.value.label.., sep = "~~~")),
                   parse = TRUE) +
      theme_minimal(base_size = 15) +  # Use minimal theme for a light grey background
      theme(
        panel.grid.major = element_line(color = "white"),   # Set major gridlines to white
        panel.grid.minor = element_line(color = "white"),   # Set minor gridlines to white
        panel.background = element_rect(fill = "lightgrey") # Light grey background
      ) +
      labs(title = paste("Scatter Plot of", names(phenotypes)[i], "vs", names(phenotypes)[j]),
           x = names(phenotypes)[i],
           y = names(phenotypes)[j])
    
    # Save the plot
    ggsave(paste0("plots/", names(phenotypes)[i], "_vs_", names(phenotypes)[j], ".png"), plot = p)
  }
}
