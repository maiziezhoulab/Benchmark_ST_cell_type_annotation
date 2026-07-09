library(GSVA)
library(GSEABase)
library(zellkonverter)
reactome <- getGmt("/Users/yuling_zhu/Downloads/msigdb_v2026.1.Hs_files_to_download_locally/msigdb_v2026.1.Hs_GMTs/c2.cp.reactome.v2026.1.Hs.symbols.gmt")
reactome_list <- geneIds(reactome)
Markes <- read.csv('/Users/yuling_zhu/Downloads/top50_markers.csv')
genes_by_group <- split(Markes$names, Markes$group)
sce <- readH5AD("/Users/yuling_zhu/Downloads/spatial_data.h5ad")
seurat_obj <- as.Seurat(sce, counts = "X", data = "X") 
counts <- GetAssayData(
  seurat_obj,
  assay = "originalexp",
  layer = "counts"
)
############# for pathway analysis 
ssgsea_param <- ssgseaParam(
  expr = as.matrix(counts),
  geneSets = reactome_list 
)
reactome_enrichscore <- gsva(ssgsea_param)
idx <- order(rowMeans(reactome_enrichscore), decreasing = T)
reactome_enrichscore <- reactome_enrichscore[idx,]
write.table(reactome_enrichscore, file = "/Users/yuling_zhu/Downloads/GSVA_Reactome_Human.txt", quote = FALSE, sep = "\t")
###################### Gene set enrichment  (cell type * cell)
ssgsea_param <- ssgseaParam(
  expr = as.matrix(counts),
  geneSets = genes_by_group
)

reactome_enrichscore <- gsva(ssgsea_param)
idx <- order(rowMeans(reactome_enrichscore), decreasing = T)
reactome_enrichscore <- reactome_enrichscore[idx,]
write.table(reactome_enrichscore, file = "/Users/yuling_zhu/Downloads/GSVA_Reactome_Human.txt", quote = FALSE, sep = "\t")
 
