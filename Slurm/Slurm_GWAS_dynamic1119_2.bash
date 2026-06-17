#!/bin/bash
#! Specify your job name
#SBATCH -J dynamic1119_2
#SBATCH -A bionf
#SBATCH --cpus-per-task=1
#SBATCH --time=100:00:00
#SBATCH --mem=16G
#SBATCH -p production
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=sw983@cam.ac.uk
#SBATCH --no-requeue
#SBATCH -e slurm-%A_%a.error
#SBATCH -o slurm-%A_%a.out

# Function to run genome_scan for a given phenotype
run_genome_scan() {
    phenotype=$1
    echo "Running genome_scan for phenotype: $phenotype"
    genome_scan -f dynamic_blues_line2.txt -p "$phenotype" -n 1000 -t 0.01 -w "dynamic1119_2/"
    if [ $? -ne 0 ]; then
        echo "genome_scan failed for phenotype: $phenotype"
    fi
}

export -f run_genome_scan

# Extract the header line from the phenotype file to get the list of phenotypes
phenotypes=($(head -n 1 dynamic_blues_line2.txt | awk '{for (i=2; i<=NF; i++) print $i}'))

# Determine which phenotype to process based on the job array index
phenotype_index=$((SLURM_ARRAY_TASK_ID - 1))
phenotype=${phenotypes[$phenotype_index]}

# Ensure the phenotype is not empty
if [ -z "$phenotype" ]; then
    echo "No phenotype found for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID. Please check the dynamic_blues_line2.txt file."
    exit 1
fi

# Debugging: Print the selected phenotype
echo "Processing phenotype: $phenotype (Index: $phenotype_index)"

# Additional debugging: Check if phenotype is valid
if ! grep -q "$phenotype" dynamic_blues_line2.txt; then
    echo "Phenotype $phenotype not found in the input file"
    exit 1
fi

# Run the genome_scan for the specific phenotype
run_genome_scan "$phenotype"

# List the contents of the home directory
ls -F ~/ > content_of_my_home_directory.txt
