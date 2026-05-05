library(reticulate)
library(Seurat)
library(anndata)
library(scuttle)
library(SingleCellExperiment)
library(SummarizedExperiment)
library(SingleR)
library(scater)
library(zellkonverter)
library(dplyr)
library(peakRAM)
query <- c('L', 'R') 
time_point <-  c('Sham', 'Hour4', 'Hour12', 'Day2', 'Day14', 'Week6')
perf_list <- list()
for ( i in 1:length(time_point)){
  for (j in 1:length(query)){
        sce <- readH5AD('/maiziezhou_lab2/yuling/datasets/Kidney/snRNA_cleaned.h5ad')
        #sce <- readH5AD(paste0("/maiziezhou_lab2/yuling/datasets/Kidney/snRNA-seq/time_", time_point[i] ,".h5ad"))
        seurat_obj <- as.Seurat(sce, counts = "X", data = "X") 
        unique_Section <- unique(seurat_obj@meta.data$n_section)
        # load single cell data 
        per_section_ref <- seurat_obj
        per_section_ref@meta.data <- per_section_ref@meta.data %>%
        mutate(name = case_when(
            name %in% c("MTAL", "CTAL1", "CTAL2") ~ "TAL",
            name %in% c("CNT", "DCT-CNT") ~ "CNT",
            name %in% c("EC1", "EC2") ~ "EC",
            name %in% c("PC1", "PC2") ~ "PC",
            name %in% c("NewPT1") ~ "Inj_PT",
            name %in% c("NewPT2") ~ "FR_PT",
            name %in% c("Mø", "Tcell") ~ "Immune",
            TRUE ~ name 
        ))
        counts <- per_section_ref@assays$originalexp@counts
        cell_metadata <- as.data.frame(per_section_ref@meta.data)
        RNAseurat <- CreateSeuratObject(counts = counts, meta.data = cell_metadata)
        RNAseurat_norm <- NormalizeData(RNAseurat, normalization.method = "LogNormalize", scale.factor = 10000)
        RNAseurat_norm  <- ScaleData(RNAseurat_norm , assay = "RNA")
        ##############
        sce <- readH5AD(paste0("/maiziezhou_lab2/yuling/datasets/Kidney/Xenium/time_", time_point[i], query[j], ".h5ad"))
        seurat_obj <- as.Seurat(sce, counts = "X", data = "X") 
 
        counts <- seurat_obj@assays$originalexp@counts
        #counts <- as.matrix(counts)
        # Extract cell metadata
        #cell_metadata <- as.data.frame(ST$obs)
        cell_metadata <- as.data.frame(seurat_obj@meta.data)
        STseurat <- CreateSeuratObject(counts = counts, meta.data = cell_metadata)
        STseurat_norm <- NormalizeData(STseurat, normalization.method = "LogNormalize", scale.factor = 10000)
        STseurat_norm  <- ScaleData(STseurat_norm , assay = "RNA")
        ref_data <- as.SingleCellExperiment(RNAseurat_norm, assay = "RNA")
        merfish_data <- as.SingleCellExperiment(STseurat_norm, assay = "RNA")
        cell_type_column <- "name" 

        # Extract cell type labels
        if (cell_type_column %in% colnames(RNAseurat_norm@meta.data)) {
        cell_types <- RNAseurat_norm@meta.data[[cell_type_column]]
        print(paste("Using cell type column:", cell_type_column))
        print(paste("Number of unique cell types:", length(unique(cell_types))))
        print("Cell type distribution:")
        print(table(cell_types))
        } else {
        stop("Cell type annotation column not found. Please check your metadata column names.")
        }
        print("Running SingleR...")
        peak_res <- peakRAM::peakRAM({
        pred_clust <- SingleR(
            test = merfish_data,
            ref = ref_data,
            labels = cell_types,
            de.method = "wilcox"
        )
        })
        time_sec  <- peak_res$Elapsed_Time_sec[1]
        peak_mem  <- peak_res$Peak_RAM_MiB[1]
            # run singleR
        #pred_clust <- SingleR(
        #test = merfish_data,
        #ref = ref_data,
        #labels = cell_types,
        #de.method = "wilcox"
        #)
        # Display results summary
        print("SingleR prediction summary:")
        print(table(pred_clust$labels))
        names(pred_clust$labels) <- colnames(STseurat_norm)
        STseurat_norm$SingleR_labels <- pred_clust$labels
        #names(pred_clust$scores) <- colnames(STseurat_norm)
        #STseurat_norm$SingleR_scores <- pred_clust$scores
        print("SingleR analysis completed successfully!")
        write.csv(pred_clust, file = paste0("/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/SingleR/Kidney_all_output/", time_point[i], query[j], "_Kidney.csv"), row.names = TRUE)
        perf_list[[length(perf_list) + 1]] <- data.frame(
             
            time_point = paste0(i,j),
            runtime_sec = time_sec,
            peak_memory_MiB = peak_mem
            )
    }
}
perf_all <- do.call(rbind, perf_list)

write.csv(
  perf_all,
  file.path("/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/SingleR/Kidney_all_output", "runtimeSec_memoryMiB.csv"),
  row.names = FALSE
)