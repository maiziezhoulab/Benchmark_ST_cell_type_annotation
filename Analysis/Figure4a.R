library(igraph)
library(dplyr)
library(tidyr)
library(stringr)
library(tibble)
library(ggraph)
library(ggplot2)
# conda activate hest
big_csv  <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/spatialID/0503_F4_C_metric/big_group_accuracy_per_series.csv"
sub_csv  <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/spatialID/0503_F4_C_metric/subgroup_accuracy_per_series.csv"
id_csv   <- "/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/spatialID/0503_F4_C_metric/identity_accuracy_per_series.csv"

big_df <- read.csv(big_csv, check.names = FALSE)
sub_df <- read.csv(sub_csv, check.names = FALSE)
id_df  <- read.csv(id_csv , check.names = FALSE)

big_row <- big_df[17, , drop = FALSE]
sub_row <- sub_df[17, , drop = FALSE]
id_row  <- id_df [17, , drop = FALSE]

big_long <- as_tibble(big_row) %>% pivot_longer(cols = -series, names_to = "name", values_to = "value")
sub_long <- as_tibble(sub_row) %>% pivot_longer(cols = -series, names_to = "name", values_to = "value")

id_long <- as_tibble(id_row) %>%
  pivot_longer(cols = -series, names_to = "colkey", values_to = "value") %>%
  separate(colkey, into = c("group_name", "identity"), sep = "::", fill = "right", extra = "merge") %>%
  mutate(identity = trimws(identity))
big_names <- big_long$name
edges_root <- data.frame(from = "Root", to = big_names, stringsAsFactors = FALSE)

escape_regex <- function(x) { str_replace_all(x, "([\\^$.|?*+()\\[\\]{}\\\\])", "\\\\\\1") }

edges_children <- lapply(big_names, function(g) {
  patt <- paste0("^", escape_regex(g), "\\(")
  subs <- sub_long %>% filter(str_detect(name, patt)) %>% pull(name)
  if (length(subs) == 0) return(NULL)
  data.frame(from = rep(g, length(subs)), to = subs, stringsAsFactors = FALSE)
}) %>% bind_rows()

GN1 <- "MV+M+VH (intermedial→ventral)"
GN2 <- "Dorsal excitatory"
GN3 <- "Dorsal inhibitory"
GN4 <- "Dorsal Maf"
GN5 <- "Cholinergic"

SUBGROUP_MAP <- list(
  # gid = 1
  `MV+M+VH (intermedial→ventral)(1)` = c("M-ex-Neurod2","MV-ex-Syt2"),
  `MV+M+VH (intermedial→ventral)(2)` = c("M-ex-Vsx2","M-ex-Vsx2/Shox2","MV-ex-Shox2"),
  `MV+M+VH (intermedial→ventral)(3)` = c("M-in-Tfap2b","MV-in-Chrna2","MV-in-Esrrb","MV-in-Gabra1",
                                         "MV-in-Gm26673","MV-in-Sema5b","VH-in-Chat"),
  # gid = 2
  `Dorsal excitatory(1)` = c("DM-ex-Zfhx3","DH-ex-Cpne4","DH-ex-Gpr83","DH-ex-Grp"),
  `Dorsal excitatory(2)` = c("DH-ex-Nmu/Tac2","DH-ex-Tac2"),
  `Dorsal excitatory(3)` = c("DH-ex-Prkcg/Cck","DH-ex-Prkcg/Nts","DH-ex-Prkcg/Rxfp1"),
  `Dorsal excitatory(4)` = c("DH-ex-Reln","DH-ex-Reln/Nmur2","DH-ex-Reln/Npff"),
  `Dorsal excitatory(5)` = c("DH-ex-Sox5","DH-ex-Sox5/Tac1"),
  # gid = 3
  `Dorsal inhibitory(1)` = c("DH-in-Cdh3","DH-in-Kcnip2","DH-in-Klhl14","DH-in-Rorb"),
  `Dorsal inhibitory(2)` = c("DH-in-Npy","DH-in-Npy2r"),
  `Dorsal inhibitory(3)` = c("DH-in-Pdyn","DH-in-Pdyn/Gal"),
  # gid = 4
  `Dorsal Maf(1)` = c("DH-ex-Rreb1"),
  `Dorsal Maf(2)` = c("DH-ex-Maf/Cck","DH-ex-Maf/Cpne4","DH-ex-Maf/Slc17a8"),
  # gid = 5
  `Cholinergic(1)` = c("alpha motoneuron","gamma motoneuron","cholinergic interneuron","visceral motoneuron")
)
ID2PARENT <- enframe(SUBGROUP_MAP, name = "parent", value = "ids") %>%
  unnest_longer(ids) %>%
  transmute(identity = ids, parent = parent)

id_long_join <- id_long %>%
  left_join(ID2PARENT, by = "identity") %>%
  filter(!is.na(parent)) %>%
  rowwise() %>%
  filter(str_starts(parent, paste0("^", escape_regex(group_name)))) %>%
  ungroup()

edges_identity <- id_long_join %>%
  transmute(from = parent, to = paste0(group_name, "::", identity))

edges <- bind_rows(edges_root, edges_children, edges_identity)

node_big  <- big_long %>% select(name, value)
node_sub  <- sub_long %>% select(name, value)
node_id   <- id_long_join %>% transmute(name = paste0(group_name, "::", identity), value = value)

node_root <- tibble(name = "Root", value = sum(node_big$value, na.rm = TRUE))
node_values <- bind_rows(node_root, node_big, node_sub, node_id) %>% distinct(name, .keep_all = TRUE)
g <- graph_from_data_frame(edges, directed = TRUE, vertices = node_values)

V(g)$children <- degree(g, mode = "out")
V(g)$width  <- pmax(1.2, 0.35 * str_length(V(g)$name) + 0.45 * V(g)$children)
V(g)$height <- 0.9
V(g)$label <- ifelse(V(g)$name == "Root", "Root",
                     paste0(V(g)$name, "\n", sprintf("%.3f", V(g)$value)))
V(g)$size <- ifelse(V(g)$name == "Root", 38, 20)

lay <- create_layout(g, layout = "tree")
x_stretch <- 12  
y_stretch <- 12  
lay$x <- lay$x * x_stretch
lay$y <- lay$y * y_stretch
#tmp <- lay$x
#lay$x <- -lay$y
#lay$y <-  tmp

lab <- ifelse(lay$name == "Root", "Root",
              ifelse(str_detect(lay$name, "::"),
                     paste0(stringr::str_wrap(str_replace(lay$name, "^.*::", ""), width = 18), "\n", sprintf("%.3f", lay$value)),
                     paste0(stringr::str_wrap(lay$name, width = 18), "\n", sprintf("%.3f", lay$value))))

# save as pdf 
pdf("/maiziezhou_lab2/yuling/MouseSpinal/label_transfer/metric/tree_spread_SpatialID_17_0503_F4_C_0210.pdf",
    width = 66, height = 12, bg = "white")
ggraph(lay) +
  geom_edge_link(arrow = arrow(length = unit(2, "mm")), width = 0.6, alpha = 0.55) +
  geom_node_point(size = 16, colour = "lightblue") +
  geom_node_text(aes(label = lab), repel = FALSE, size = 4, vjust = 1.5, lineheight = 0.95) +
  ggtitle("Accuracy Tree for Spatial-ID") +
  coord_cartesian(clip = "off") +
  theme_void(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5, size = 20, face = "bold"),
        plot.margin = margin(20, 60, 20, 60))
dev.off()
# 
unmatched <- anti_join(id_long, ID2PARENT, by = "identity")
if (nrow(unmatched) > 0) {
  message("Unmatched identities (not drawn): ",
          paste(unique(unmatched$identity), collapse = "; "))
}