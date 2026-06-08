library(ggplot2)
library(dplyr)
library(tidyr)
library(stringr)
library(tibble)

# ===================== Path =====================
method_dirs <- list(
  SpatialID = "/maiziezhou_lab2/yuling/MouseSpinal/Project/spatialID/0503_F4_C_output",
  DestVI = "/maiziezhou_lab2/yuling/MouseSpinal/Project/DestVI/0503_F4_C_output",
  RCTD = "/maiziezhou_lab2/yuling/MouseSpinal/Project/RCTD/0503_F4_C_output",
  CARD = "/maiziezhou_lab2/yuling/MouseSpinal/Project/CARD/0503_F4_C_output",
  Cell2location = "/maiziezhou_lab2/yuling/MouseSpinal/Project/cell2location/0503_F4_C_output",
  GraphST = "/maiziezhou_lab2/yuling/MouseSpinal/Project/GraphST/0503_F4_C_output",
  scANVI = "/maiziezhou_lab2/yuling/MouseSpinal/Project/scVI/0503_F4_C_output",
  Seurat = "/maiziezhou_lab2/yuling/MouseSpinal/Project/Seurat/0503_F4_C_output",
  SingleR = "/maiziezhou_lab2/yuling/MouseSpinal/Project/SingleR/0503_F4_C_output",
  SpatialDWLS = "/maiziezhou_lab2/yuling/MouseSpinal/Project/spatialDWLS/0503_F4_C_output",
  SPOTlight = "/maiziezhou_lab2/yuling/MouseSpinal/Project/spotlight/0503_F4_C_output",
  TACCO = "/maiziezhou_lab2/yuling/MouseSpinal/Project/tacco/0503_F4_C_output",
  Tangram = "/maiziezhou_lab2/yuling/MouseSpinal/Project/Tangram/0503_F4_C_output"
)

# ===================== read data =====================
read_method_data <- function(method_name, method_dir) {
  tryCatch({
    big_df <- read.csv(file.path(method_dir, "big_group_accuracy_per_series.csv"), check.names = FALSE)
    sub_df <- read.csv(file.path(method_dir, "subgroup_accuracy_per_series.csv"), check.names = FALSE)
    id_df <- read.csv(file.path(method_dir, "identity_accuracy_per_series.csv"), check.names = FALSE)
    
    list(
      big = big_df[17, , drop = FALSE],
      sub = sub_df[17, , drop = FALSE],
      id = id_df[17, , drop = FALSE]
    )
  }, error = function(e) {
    warning(paste("Failed to read data for", method_name, ":", e$message))
    NULL
  })
}

all_method_data <- lapply(names(method_dirs), function(m) {
  cat("Reading data for:", m, "\n")
  read_method_data(m, method_dirs[[m]])
})
names(all_method_data) <- names(method_dirs)
all_method_data <- Filter(Negate(is.null), all_method_data)

SUBGROUP_MAP <- list(
  # gid = 1
  `MV+M+VH (intermedial→ventral)(1)` = c("M-ex-Neurod2","MV-ex-Syt2"),
  `MV+M+VH (intermedial→ventral)(2)` = c("M-ex-Vsx2","M-ex-Vsx2/Shox2","MV-ex-Shox2"),
  `MV+M+VH (intermedial→ventral)(3)` = c("M-in-Tfap2b","MV-in-Chrna2","MV-in-Esrrb","MV-in-Gabra1",
                                          "MV-in-Gm26673","MV-in-Sema5b","VH-in-Chat"),
  # gid = 2
  `Dorsal excitatory(1)` = c("DM-ex-Zfhx3","DH-ex-Cpne4","DH-ex-Gpr83","DH-ex-Grp","DH-ex-Rreb1"),
  `Dorsal excitatory(2)` = c("DH-ex-Nmu/Tac2","DH-ex-Tac2"),
  `Dorsal excitatory(3)` = c("DH-ex-Prkcg/Cck","DH-ex-Prkcg/Nts","DH-ex-Prkcg/Rxfp1"),
  `Dorsal excitatory(4)` = c("DH-ex-Reln","DH-ex-Reln/Nmur2","DH-ex-Reln/Npff"),
  `Dorsal excitatory(5)` = c("DH-ex-Sox5","DH-ex-Sox5/Tac1"),
  `Dorsal excitatory(6)` = c("DH-ex-Maf/Cck","DH-ex-Maf/Cpne4","DH-ex-Maf/Slc17a8"),
  # gid = 3
  `Dorsal inhibitory(1)` = c("DH-in-Cdh3","DH-in-Kcnip2","DH-in-Klhl14","DH-in-Rorb"),
  `Dorsal inhibitory(2)` = c("DH-in-Npy","DH-in-Npy2r"),
  `Dorsal inhibitory(3)` = c("DH-in-Pdyn","DH-in-Pdyn/Gal"),
  # gid = 5
  `Cholinergic(1)` = c("alpha motoneuron","gamma motoneuron","cholinergic interneuron","visceral motoneuron")
)

# from big group to small group 
BIG_TO_SUB <- list(
  `Cholinergic` = c("Cholinergic(1)"),
  `Dorsal inhibitory` = c("Dorsal inhibitory(1)", "Dorsal inhibitory(2)", "Dorsal inhibitory(3)"),
  `Dorsal excitatory` = c("Dorsal excitatory(1)", "Dorsal excitatory(2)", "Dorsal excitatory(3)",
                          "Dorsal excitatory(4)", "Dorsal excitatory(5)", "Dorsal excitatory(6)"),
  `MV+M+VH (intermedial→ventral)` = c("MV+M+VH (intermedial→ventral)(1)",
                                       "MV+M+VH (intermedial→ventral)(2)",
                                       "MV+M+VH (intermedial→ventral)(3)")
)
prepare_method_data <- function(big_row, sub_row, id_row, method_name) {
  big_long <- as_tibble(big_row) %>%
    pivot_longer(cols = -series, names_to = "big_group", values_to = "value") %>%
    mutate(method = method_name, level = "BIG")
  
  # if none, fill it as 0
  all_big_groups <- names(BIG_TO_SUB)
  big_long <- big_long %>%
    complete(big_group = all_big_groups, fill = list(value = 0)) %>%
    mutate(method = method_name, level = "BIG", series = 1)
  
  # process subgroup 
  sub_long <- as_tibble(sub_row) %>%
    pivot_longer(cols = -series, names_to = "subgroup", values_to = "value") %>%
    mutate(method = method_name, level = "SUB")
  all_subgroups <- names(SUBGROUP_MAP)
  sub_long <- sub_long %>%
    complete(subgroup = all_subgroups, fill = list(value = 0)) %>%
    mutate(method = method_name, level = "SUB", series = 1)
  
  # process identity 
  id_long <- as_tibble(id_row) %>%
    pivot_longer(cols = -series, names_to = "colkey", values_to = "value") %>%
    separate(colkey, into = c("group_name", "identity"), sep = "::", fill = "right", extra = "merge") %>%
    mutate(identity = trimws(identity),
           method = method_name,
           level = "IDENTITY")

  ID2PARENT <- enframe(SUBGROUP_MAP, name = "parent", value = "ids") %>%
    unnest_longer(ids) %>%
    transmute(identity = ids, subgroup = parent)
  all_identities <- unlist(SUBGROUP_MAP, use.names = FALSE)
  id_complete <- tibble(
    identity = all_identities,
    method = method_name,
    level = "IDENTITY",
    series = 1
  ) %>%
    left_join(id_long %>% select(identity, value), by = "identity") %>%
    mutate(value = ifelse(is.na(value), 0, value))
  
  id_long <- id_complete %>%
    left_join(ID2PARENT, by = "identity")
  
  # find group big group for each sub group 
  SUB2BIG <- enframe(BIG_TO_SUB, name = "big_group", value = "subs") %>%
    unnest_longer(subs) %>%
    transmute(subgroup = subs, big_group = big_group)
  
  sub_long <- sub_long %>%
    left_join(SUB2BIG, by = "subgroup")
  
  id_long <- id_long %>%
    left_join(SUB2BIG, by = "subgroup")
  
  return(list(big = big_long, sub = sub_long, identity = id_long))
}

# prepare data for each tool 
all_prepared_data <- lapply(names(all_method_data), function(m) {
  prepare_method_data(
    all_method_data[[m]]$big,
    all_method_data[[m]]$sub,
    all_method_data[[m]]$id,
    m
  )
})
names(all_prepared_data) <- names(all_method_data)

# combine all of the data 
all_big <- bind_rows(lapply(all_prepared_data, function(x) x$big))
all_sub <- bind_rows(lapply(all_prepared_data, function(x) x$sub))
all_id <- bind_rows(lapply(all_prepared_data, function(x) x$identity))

big_order <- c("Cholinergic", "Dorsal inhibitory", "Dorsal excitatory",
               "MV+M+VH (intermedial→ventral)")
identity_counts <- all_id %>%
  group_by(big_group, subgroup) %>%
  summarise(n_id = n_distinct(identity), .groups = "drop")
x_pos <- 0
big_positions <- list()
sub_positions <- list()
id_positions <- list()

for (bg in big_order) {
  subs <- BIG_TO_SUB[[bg]]
  n_ids_total <- 0
  sub_start <- x_pos
  
  for (sg in subs) {
    ids <- SUBGROUP_MAP[[sg]]
    n_ids <- length(ids)
    
    # identity positions
    for (i in seq_along(ids)) {
      id_positions[[paste(bg, sg, ids[i], sep = "|")]] <- x_pos + i - 0.5
    }
    
    # subgroup position (center of its identities)
    sub_positions[[paste(bg, sg, sep = "|")]] <- x_pos + n_ids / 2
    x_pos <- x_pos + n_ids
    n_ids_total <- n_ids_total + n_ids
  }
  
  # big group position (center of all its identities)
  big_positions[[bg]] <- sub_start + n_ids_total / 2
}
all_big <- all_big %>%
  rowwise() %>%
  mutate(x_pos = big_positions[[big_group]]) %>%
  ungroup()

all_sub <- all_sub %>%
  rowwise() %>%
  mutate(x_pos = sub_positions[[paste(big_group, subgroup, sep = "|")]]) %>%
  ungroup()

all_id <- all_id %>%
  rowwise() %>%
  mutate(x_pos = id_positions[[paste(big_group, subgroup, identity, sep = "|")]]) %>%
  ungroup()

#method_order <- c("spatialID", "scVI", "Seurat", "tacco", "SingleR", "RCTD", 
    #              "Tangram", "DestVI", "spatialDWLS", "spotlight", "CARD", 
       #           "cell2location", "GraphST")
method_order <- c("SpatialID", "scANVI", "Seurat", "TACCO", "SingleR", "RCTD", 
                  "Tangram", "DestVI", "SpatialDWLS", "SPOTlight", "CARD", 
                  "Cell2location", "GraphST")

available_methods <- names(all_method_data)
method_order <- method_order[method_order %in% available_methods]

cat("\n=== Method order (top to bottom) ===\n")
for (i in seq_along(method_order)) {
  cat(i, ". ", method_order[i], "\n", sep = "")
}

n_methods <- length(method_order)
band_gap <- 0      
band_height <- 0.8 

rows_per_method <- 3
method_spacing <- 0.3  
offset_ID <- 0
offset_SUB <- 1
offset_BIG <- 2
method_order_1 <- c("Spatial-ID", "scANVI", "Seurat", "TACCO", "SingleR", "RCTD", 
                  "Tangram", "DestVI", "SpatialDWLS", "SPOTlight", "CARD", 
                  "Cell2location", "GraphST")
method_y_base <- setNames(
  seq(n_methods, 1, by = -1) * (rows_per_method + method_spacing) - method_spacing,
  method_order
)
all_big <- all_big %>%
  filter(method %in% method_order) %>%
  mutate(y_pos = method_y_base[method] + offset_BIG)

all_sub <- all_sub %>%
  filter(method %in% method_order) %>%
  mutate(y_pos = method_y_base[method] + offset_SUB)

all_id <- all_id %>%
  filter(method %in% method_order) %>%
  mutate(y_pos = method_y_base[method] + offset_ID)

big_widths <- all_id %>%
  group_by(big_group) %>%
  summarise(n_ids = n_distinct(identity), .groups = "drop")

big_tile_data <- all_big %>%
  left_join(big_widths, by = "big_group") %>%
  mutate(xmin = x_pos - n_ids / 2,
         xmax = x_pos + n_ids / 2,
         ymin = y_pos - band_height/2,
         ymax = y_pos + band_height/2)

sub_tile_data <- all_sub %>%
  group_by(method, big_group, subgroup, y_pos, x_pos) %>%
  summarise(value = first(value), .groups = "drop") %>%
  left_join(identity_counts, by = c("big_group", "subgroup")) %>%
  mutate(xmin = x_pos - n_id / 2,
         xmax = x_pos + n_id / 2,
         ymin = y_pos - band_height/2,
         ymax = y_pos + band_height/2)
id_tile_data <- all_id %>%
  mutate(xmin = x_pos - 0.5,
         xmax = x_pos + 0.5,
         ymin = y_pos - band_height/2,
         ymax = y_pos + band_height/2)

y_axis_breaks <- method_y_base + 1  
y_axis_labels <- names(method_y_base)

p <- ggplot() +
  # BIG layer 
  geom_rect(data = big_tile_data,
            aes(xmin = xmin, xmax = xmax,
                ymin = ymin, ymax = ymax,
                fill = value), color = NA) +
  geom_text(data = big_tile_data,
            aes(x = x_pos, y = y_pos, label = sprintf("%.2f", value)),
            size = 3, color = "white", fontface = "bold") +
  
  # SUB layer 
  geom_rect(data = sub_tile_data,
            aes(xmin = xmin, xmax = xmax,
                ymin = ymin, ymax = ymax,
                fill = value), color = NA) +
  geom_text(data = sub_tile_data,
            aes(x = x_pos, y = y_pos, label = sprintf("%.2f", value)),
            size = 2.8, color = "white", fontface = "bold") +
  
  # IDENTITY layer 
  geom_rect(data = id_tile_data,
            aes(xmin = xmin, xmax = xmax,
                ymin = ymin, ymax = ymax,
                fill = value), color = NA) +
  geom_text(data = id_tile_data,
            aes(x = x_pos, y = y_pos, label = sprintf("%.2f", value)),
            size = 2.5, color = "white") +
  
  scale_fill_viridis_c(limits = c(0, 1), option = "viridis", name = "Accuracy") +
  scale_y_continuous(breaks = y_axis_breaks,
                     labels = y_axis_labels,
                     expand = c(0.01, 0.01)) +
  scale_x_continuous(expand = c(0.01, 0.01)) +
  labs(x = "", y = "") +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.y = element_text(size = 12, face = "bold", hjust = 1),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    panel.grid = element_blank(),
    legend.position = "right",
    plot.margin = margin(10, 10, 10, 10)
  )
output_dir <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/metric"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

output_file <- file.path(output_dir, "Hierarchical_heatmap_13methods_0503_F4_C.pdf")
ggsave(output_file,
       plot = p, 
       width = 20, 
       height = n_methods * 1.2, 
       device = pdf,
       family = "Helvetica",
       useDingbats = FALSE,
       bg = "white")

output_svg_file <- file.path(output_dir, "Hierarchical_heatmap_13methods_0503_F4_C.svg")
if (requireNamespace("svglite", quietly = TRUE)) {
  ggsave(output_svg_file,
         plot = p,
         width = 20,
         height = n_methods * 1.2,
         device = svglite::svglite,
         system_fonts = list(sans = "Helvetica"),
         bg = "white")
} else {
  warning("Package 'svglite' is not installed; SVG export was skipped.")
}

cat("\n=== SUCCESS ===\n")
cat("Heatmap saved to:", output_file, "\n")
if (file.exists(output_svg_file)) {
  cat("Editable SVG saved to:", output_svg_file, "\n")
}
cat("Total methods:", n_methods, "\n")