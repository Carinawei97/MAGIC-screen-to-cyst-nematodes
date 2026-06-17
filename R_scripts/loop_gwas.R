# Load necessary libraries
lib.list <- c("stringr", "ggplot2")
lapply(lib.list, library, character.only = TRUE)

# Set the working directory
setwd("/Users/pro/Desktop/Dynamic")

# Load functions to run mixed model
source("mixed.model.functions.r")

# Define the phenotype columns to be analyzed
phenotype_columns <- c('fastest_dpi', 'fastest_rate', 'fastest_size', 'final_day_of_growth', 
                       'first_day_of_growth', 'gen_rate', 'length', 'max_size')

# Column name with sample names (must match sample names of genotype data)
id.column <- "SUBJECT.NAME"

# Phenotype file location
phenfile <- "clean2_dynamic_gwas.txt"

# Location of the pruned traw file
pruned_traw_file <- "magic.traw.RData"  # Imputed MAGIC RIL genotypes in traw format, after LD pruning

# Parameters
n.perm <- 1000  # Number of permutations to run to establish significance thresholds
alpha <- c(0.05, 0.05)  # Genomewide significance levels

# Make genetic relationship matrix
pruned_grm_file <- str_replace(pruned_traw_file, ".traw", ".grm")  # Default output location for GRM
if (!file.exists(pruned_grm_file)) {
  make_grm(traw_file = pruned_traw_file, output_filename = pruned_grm_file)
}

# Loop through each phenotype column
for (phenotype in phenotype_columns) {
  
  # Create output directory for the current phenotype
  out_dir <- paste0("OUTPUT/", phenotype, "/")
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Run SNP associations for the current phenotype
  snp.output <- mixed.model.snps(phenfile, phenotype, id.column, pruned_grm_file, pruned_traw_file, n.perm, covar = NULL, rename_CHR = TRUE, rename_samples = TRUE)
  
  # Save SNP output
  save(snp.output, file = paste0(out_dir, "snp.output.RData"))
  
  # Write logP values to CSV
  output_file_path <- file.path(out_dir, "logP.csv")
  write.csv(snp.output[["logP"]], file = output_file_path)
  
  # Load plotting functions
  source("plot.functions.r")
  
  # Produce a Manhattan plot for the current phenotype
  snp.manhattan.plot(snp.output, snp.manhattan.filename = paste0(out_dir, "snp.manhattan.pdf"), alpha = c(0.05, 0.05))
}