## Generate R MDS coordinates for comparison notebook
library(edgeR)

# Load data
counts <- read.delim("countdata.tsv", row.names=1)
factors <- read.delim("sampleinfo.tsv")

# Create DGEList
group <- factor(factors$CellTypeStatus)
d <- DGEList(counts=counts, group=group)

# Filter (CPM > 0.5 in >= 2 samples)
cpm_vals <- cpm(d)
keep <- rowSums(cpm_vals > 0.5) >= 2
d <- d[keep, , keep.lib.sizes=FALSE]

# TMM normalization
d <- calcNormFactors(d)

# Extract MDS coordinates
mds_obj <- plotMDS(d, plot=FALSE)

# Save
write.csv(data.frame(
  SampleName = colnames(d),
  MDS1 = mds_obj$x,
  MDS2 = mds_obj$y,
  var.explained.1 = mds_obj$var.explained[1],
  var.explained.2 = mds_obj$var.explained[2]
), "R_mds_coordinates.csv", row.names=FALSE)

cat("R MDS coordinates saved to R_mds_coordinates.csv\n")
cat("Samples:", ncol(d), "\n")
cat("Genes:", nrow(d), "\n")
