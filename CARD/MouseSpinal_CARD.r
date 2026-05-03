library(Seurat)
library(SeuratData)
library(ggplot2)
library(cowplot)
library(patchwork)
library(dplyr)
library(CARD)
library(zellkonverter)
library(peakRAM)
library(tibble)

# ----------------------------
# Output paths
# ----------------------------
out_prop <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/CARD/0503_F4_C_output/Proportion_0503_F4_C_CARD.csv"
out_log  <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/CARD/0503_F4_C_output/runtimeSec_memoryMiB.csv"

# ----------------------------
# Measure time + peak memory
# ----------------------------
peak_res <- peakRAM({

  time_res <- system.time({

    # ---- Load data
    sce <- readH5AD("/maiziezhou_lab2/yuling/MERFISH_spinal_cord_resolved_0718.h5ad")
    seurat_obj <- as.Seurat(sce, counts = "X", data = "X")

    unique_Section <- unique(seurat_obj@meta.data$Section.ID)
    selected_0503 <- grep("^0503", unique_Section, value = TRUE)
    selected_0503_clean <- setdiff(selected_0503, "0503_nan_nan")
    selected_0503_1 <- setdiff(selected_0503_clean, "0503_F4_C")

    # ---- Single-cell reference
    total_data <- subset(
      seurat_obj,
      subset = Section.ID %in% selected_0503_1
    )

    # ---- Spatial query
    per_section_spatial <- subset(
      seurat_obj,
      subset = Section.ID == "0503_F4_C"
    )

    spatial_count <- per_section_spatial@assays$originalexp$data

    # ---- Spatial coordinates
    meta <- per_section_spatial[[]]
    centers <- meta[, c("center_x", "center_y"), drop = FALSE]
    spatial_location <- cbind(cell_id = rownames(centers),
                 setNames(centers, c("x", "y")))
    rownames(spatial_location) <- NULL
    spatial_location <- tibble::column_to_rownames(spatial_location, var = "cell_id")

        # ---- scRNA-seq data
    sc_count <- total_data@assays$originalexp$data
    sc_meta <- total_data@meta.data %>%
      rename(cellType = MERFISH.cell.type.annotation)

    sc_meta$sampleInfo <- "sample1"

    # ---- CARD object
    CARD_obj <- createCARDObject(
      sc_count = sc_count,
      sc_meta = sc_meta,
      spatial_count = spatial_count,
      spatial_location = spatial_location,
      ct.varname = "cellType",
      ct.select = unique(sc_meta$cellType),
      sample.varname = "sampleInfo",
      minCountGene = 0,
      minCountSpot = 0
    )

    # ---- CARD deconvolution
    CARD_obj <- CARD_deconvolution(CARD_object = CARD_obj)

    # ---- Save proportions
    write.csv(
      CARD_obj@Proportion_CARD,
      out_prop,
      quote = FALSE
    )

  }) # system.time

}) # peakRAM

# ----------------------------
# Collect runtime + memory
# ----------------------------
runtime_sec <- time_res["elapsed"]
peak_mem_mb <- max(peak_res$Peak_RAM_Used_MiB)

summary_df <- data.frame(
  runtime_sec = as.numeric(runtime_sec),
  peak_memory_MiB = as.numeric(peak_mem_mb)
)

write.csv(summary_df, out_log, row.names = FALSE)

print(summary_df)
