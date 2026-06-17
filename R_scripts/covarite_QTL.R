##QTL analysis with covariarte SIYUAN WEI 2023-04-28
# install the remotes package if not installed already
if (!("remotes" %in% rownames(installed.packages()))){
  install.packages("remotes")
}

# Install atMAGIC
remotes::install_github("tavareshugo/atMAGIC",force = TRUE)
# make sure the remotes package is installed
remotes::install_github("tavareshugo/qtl2helper",force = TRUE)

library(qtl2)
library(qtl2helper)
library(atMAGIC)
library(ggplot2)
library(dplyr)
library(data.table)
library(ggpmisc)
# Load the data to the environment
data("kover2009")
setwd("/Users/pro/Desktop/Distribution.v1")
# Look at its documentation for more details
?kover2009

pheno <- read.table("trial.txt",
                    header = TRUE)

chr<- read.csv("wide_BLUEs_size.csv",header = T)
write.table(chr,'wide_BLUEs_size.txt',row.names = F,sep ="\t", quote = F, col.names = T)
pheno <- read.table("wide_BLUEs_size.txt",header = TRUE)

#pheno$ratio <- pheno$number_64dpi/pheno$number_41dpi
head(pheno)
str(pheno)
pheno$nd_gen <- as.numeric(pheno$nd_gen)


# Add to kover2009 - retain only individuals with phenotype data
kover2009 <- add_pheno(kover2009, pheno, 
                       idcol = "SUBJECT.NAME",
                       retain_all = FALSE)
# Calculate genotype probabilities
kover2009_probs <- calc_genoprob(kover2009)

# Normal scan ----
# scan for all traits in the object
trait_scan <- scan1(kover2009_probs, 
                    pheno = kover2009$pheno)
# visualise the scan for one of the traits
plot(trait_scan, kover2009$pmap, lodcolumn = "a_for_axplusb_steepest_point")
# Run permutations for genome-wide threshold
scan_perm <- scan1perm(kover2009_probs, kover2009$pheno, n_perm = 1000)
scan_threshold <- summary(scan_perm)
scan_threshold
find_peaks(trait_scan, kover2009$pmap, threshold = scan_threshold)
write.csv(trait_scan,'MAGIC_SNP_GWAS_size.csv')
## Trait Covariate
#plot(kover2009$pheno[, "root"], 
  #   kover2009$pheno[, "cystnum"])
df <- data.table(kover2009$pheno)

ggplot(df, aes(x = ratio, y = fastest )) +
  geom_point() +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE) +
  stat_poly_eq(formula = x ~ y, 
               aes(label = paste(..eq.label.., ..rr.label.., ..p.value.label.., sep = "~~~")), 
               parse = TRUE)

# Run a QTL scan
kover2009_scan1 <- scan1(kover2009_probs, kover2009$pheno[,"a_for_axplusb_steepest_point", drop = F])
plot(kover2009_scan1, kover2009$pmap, lodcolumn = "a_for_axplusb_steepest_point")

# Run a QTL scan with phenotypic co-variate
plot(trait_scan, kover2009$pmap, lodcolumn = "a_for_axplusb_steepest_point")
kover2009_scan3 <- scan1(kover2009_probs, kover2009$pheno[,"a_for_axplusb_steepest_point", drop = F], 
                         addcovar = kover2009$pheno[,"b_for_axplusb_steepest_point"])
plot(kover2009_scan3, kover2009$pmap, lodcolumn = "a_for_axplusb_steepest_point", add=T, col="blue")
#scan_perm <- scan1perm(kover2009_probs,  kover2009$pheno[,"size_41dpi", drop = F],
#                       addcovar = kover2009$pheno[,"number_64dpi"],
#                       n_perm = 1000)
#scan_threshold <- summary(scan_perm)

#find_peaks(kover2009_scan3, map = kover2009$pmap, 
#           threshold = scan_threshold)
write.csv(kover2009_scan3,'cystratio_co_cystnum.csv')

## Marker Covariate
plot(trait_scan, kover2009$pmap, "number_41dpi")
find_peaks(trait_scan, kover2009$pmap, scan_threshold)
marker_prob <- pull_genoprobpos(kover2009_probs, 
                                map = kover2009$pmap,
                                chr = 5, pos =8240542)
test <- pull_genoprobpos(kover2009_probs, 
                         map = kover2009$pmap,
                         chr = 5)
# scan with marker covariate
scan_marker_covar <- scan1(kover2009_probs, 
                           pheno = kover2009$pheno[, "a_for_axplusb_steepest_point", drop = FALSE], 
                           addcovar = b_for_axplusb_steepest_poin)
plot(scan_marker_covar, kover2009$pmap, lodcolumn = "b_for_axplusb_steepest_point",add=T, col="pink")

### SNP Covariate
# find what the marker index is
marker_idx <- which(kover2009$pmap[[5]] == 8240542)
# extract the genotype
snp_covar <- kover2009$geno[[5]][, marker_idx, drop = FALSE]
# convert it to binary
snp_covar[, 1] <- as.numeric(snp_covar[, 1] == 3)
head(snp_covar)
# scan with SNP marker covariate
scan_snp_covar <- scan1(kover2009_probs, 
                        pheno = kover2009$pheno[, "number_41dpi", drop = FALSE], 
                        addcovar = snp_covar)
plot(scan_snp_covar, kover2009$pmap, lodcolumn = "number_41dpi",add=T, col="dark grey")

#Interesting: the Chr 4 peak remains the same. 
#Is it surprising? 
#  To answer this, it helps to look at the allelic effect at the locus. 
#In addition, we colour each accession according to their SNP genotype at that locus.

# get QTL effects
chr5_eff <- scan1coef(kover2009_probs[,"5"], 
                      kover2009$pheno[,"size_41dpi"],
                      se = TRUE)
# tidy up and filter for marker of interest
chr5_eff <- chr5_eff |> 
  tidy(map = kover2009$pmap) |> 
  filter(chrom == 5 & pos == 23246248 & coef != "intercept")
# get SNP genotype for each accession at that position
acc_snp <- kover2009$founder_geno[[5]][, marker_idx, drop = FALSE]
acc_snp <- acc_snp |> 
  as_tibble(rownames = "accession") |> 
  mutate(accession = paste0(accession, accession), 
         snp = ifelse( VIN3_2942 == 3, "Alt", "Ref"))
# join and visualise
acc_snp |> 
  full_join(chr5_eff, by = c("accession" = "coef")) |> 
  mutate(accession = forcats::fct_reorder(accession, estimate)) |> 
  ggplot(aes(accession, estimate)) +
  geom_pointrange(aes(ymin = estimate - SE*2, 
                      ymax = estimate + SE*2,
                      colour = snp))






## extracting information from this package
founder_markers <- (kover2009$founder_geno)
write.csv(marker_ids[["1"]], "markers.csv")
snp_covar <- kover2009$geno[[5]][, marker_idx, drop = FALSE]
# convert it to binary
snp_covar[, 1] <- as.numeric(snp_covar[, 1] == 3)


