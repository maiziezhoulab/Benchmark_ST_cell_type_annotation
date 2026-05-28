library(dplyr)
library(tidyr)
library(ggplot2)
library(patchwork)


df_long <- read.csv("/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/Markers/pathway_ssgsea_human.csv")

ref_method <- "original_clusters"

CELL_TYPES <- c("Tumor", "CAF", "Fibroblasts", "Plasma_IgG")
PALS <- list(
  Tumor       = c("#EFF6FF", "#93C5FD", "#1D4ED8"),
  CAF         = c("#FFF7ED", "#FDBA74", "#D55E00"),
  Fibroblasts = c("#DCFCE7", "#4ADE80", "#166534"),
  Plasma_IgG  = c("#FEF9C3", "#FEF3C7", "#E6AB02")
)

build_heat_df <- function(cell_type, df_long, ref_method) {
  df_ct <- df_long %>%
    filter(.data$cell_type == .env$cell_type) %>%
    mutate(
      score   = as.numeric(score),
      method  = as.character(method),
      pathway = as.character(pathway)
    ) %>%
    filter(!is.na(score), !is.na(method), !is.na(pathway))

  if (nrow(df_ct) == 0L) {
    warning("no data: cell_type = ", cell_type)
    return(NULL)
  }

  top10_ref <- df_ct %>%
    filter(method == ref_method) %>%
    arrange(desc(score)) %>%
    slice_head(n = 10)

  if (nrow(top10_ref) == 0L) {
    warning("cannot find method == '", ref_method, "'：", cell_type)
    return(NULL)
  }

  pathway_order <- top10_ref %>%
    distinct(pathway, .keep_all = TRUE) %>%
    arrange(desc(score)) %>%
    pull(pathway)

  pathway_fixed <- pathway_order
  all_methods <- sort(unique(df_ct$method))

  heat_df <- df_ct %>%
    filter(pathway %in% pathway_fixed, method %in% all_methods) %>%
    group_by(pathway, method) %>%
    summarise(score = max(score, na.rm = TRUE), .groups = "drop") %>%
    complete(pathway = pathway_fixed, method = all_methods, fill = list(score = NA_real_))

  list(
    heat_df       = heat_df,
    pathway_order = pathway_order,
    cell_type     = cell_type
  )
}

parts <- lapply(CELL_TYPES, build_heat_df, df_long = df_long, ref_method = ref_method)
names(parts) <- CELL_TYPES
parts <- parts[!vapply(parts, is.null, logical(1))]

if (length(parts) == 0L) {
  stop("four cell types do not have valid data。")
}

method_order_by_top10_mean <- function(heat_df) {
  oc <- "original_clusters"
  rank_df <- heat_df %>%
    group_by(method) %>%
    summarise(
      m = mean(score, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(m = ifelse(is.nan(m), -Inf, m))

  all_m <- unique(as.character(heat_df$method))

  if (!oc %in% all_m) {
    return(rank_df %>% arrange(desc(m), method) %>% pull(method))
  }

  rest <- rank_df %>%
    filter(method != oc) %>%
    arrange(desc(m), method) %>%
    pull(method)

  c(oc, rest)
}

make_plot_for_ct <- function(cell_type) {
  pobj <- parts[[cell_type]]
  if (is.null(pobj)) {
    return(
      ggplot() +
        theme_void() +
        annotate("text", x = 0.5, y = 0.5, label = paste("No data:", cell_type), size = 4)
    )
  }

  heat_df <- pobj$heat_df
  pathway_order <- pobj$pathway_order

  mo <- method_order_by_top10_mean(heat_df)
  mo <- mo[mo %in% unique(heat_df$method)]

  df_norm2 <- heat_df %>%
    group_by(method) %>%
    mutate(score_z_within_method = as.numeric(scale(score))) %>%
    ungroup() %>%
    mutate(
      pathway = factor(pathway, levels = rev(pathway_order)),
      method  = factor(method, levels = mo)
    )

  pal <- PALS[[cell_type]]
  if (is.null(pal)) {
    pal <- c("#F5F5F5", "#999999", "#333333")
  }

  ggplot(df_norm2, aes(x = method, y = pathway, fill = score)) +
    geom_tile(color = "white", linewidth = 0.3) +
    scale_fill_gradientn(colors = pal, name = "score") +
    theme_minimal(base_size = 11) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
      panel.grid  = element_blank(),
      plot.title    = element_text(face = "bold", size = 11)
    ) +
    labs(
      title = paste0(cell_type, ": fixed Top10 pathways from GT"),
      x     = "",
      y     = ""
    )
}

plot_list <- lapply(names(parts), make_plot_for_ct)
names(plot_list) <- names(parts)

combined <- wrap_plots(plot_list, ncol = 2) +
  plot_annotation(
    title = "ssGSEA (Top10 pathways / cell type); x: original_clusters first, then others by mean score (10 pathways), high to low",
    theme = theme(plot.title = element_text(face = "bold", size = 12))
  )

print(combined)

out_base <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/Markers/fixedTop10_four_celltypes_2x2"
ggsave(paste0(out_base, ".png"), combined, width = 22, height = 14, dpi = 300)
ggsave(paste0(out_base, ".pdf"), combined, width = 22, height = 14, device = cairo_pdf)

message("Saved: ", out_base, ".png / .pdf")
