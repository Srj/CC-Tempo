##################################################
## Project: ost - Mice scRNAseq - Hassan Samee 
## Purpose: CellChat analysis
## Date: Nov 30, 2022
## Author: Ege Ulgen
##################################################
library(Seurat)
library(CellChat)
library(Seurat)
library(magrittr)
library(patchwork)
library(NMF)
library(ggalluvial)
devtools::install_github("renozao/NMF@devel")

# create CellChat object --------------------------------------------------
out_dir <- "plots"
day2_progenitors <- read.csv("day2_progenitors_id.csv")
seurat_obj <- readRDS("data/weinreb_seurat_scaled.rds")
seurat_obj <- subset(seurat_obj, cells = day2_progenitors$index)
#Extract Expression & Metadata
exprMat <- GetAssayData(seurat_obj, assay = "RNA", slot = "data")
meta.data <- seurat_obj@meta.data

#Create CellChat Object
cell_chat_obj <- createCellChat(object = exprMat, meta = meta.data, group.by = "Cell.type.annotation")
cell_chat_obj@DB <- CellChatDB.mouse
cell_chat_obj <- subsetData(cell_chat_obj)
cell_chat_obj <- identifyOverExpressedGenes(cell_chat_obj) %>%
  identifyOverExpressedInteractions()
cell_chat_obj <- projectData(cell_chat_obj, PPI.mouse)
cell_chat_obj <- computeCommunProb(cell_chat_obj) %>%
  filterCommunication(min.cells = 10)
cell_chat_obj <- aggregateNet(cell_chat_obj)

saveRDS(cell_chat_obj, file = paste0("data/cell_chat_obj_Day2_Progenitors.rds"))
dir.create(out_dir, recursive = TRUE)

cell_chat_obj <- readRDS("data/cell_chat_obj_Day2_Progenitors.rds")


# using the complete CellChat interactions database
cell_chat_obj@DB <- CellChatDB.mouse

rm(list = setdiff(ls(), c("cell_chat_obj", "out_dir")))

# infer the cell state-specific communications ----------------------------
future::plan("multisession", workers = 2)

# visualize interactions --------------------------------------------------
pdf(file.path(out_dir, "1.cell_cell_communications.pdf"), width = 6, height = 6)
groupSize <- as.numeric(table(cell_chat_obj@idents))
netVisual_circle(cell_chat_obj@net$count, vertex.weight = groupSize, vertex.label.cex = 0.75,
                        weight.scale = TRUE, label.edge = FALSE, title.name = "Number of interactions") 
netVisual_circle(cell_chat_obj@net$weight, vertex.weight = groupSize, vertex.label.cex = 0.75,
                        weight.scale = TRUE, label.edge = FALSE, title.name = "Interaction weights/strength")
dev.off()


pdf(file.path(out_dir, "2.communications_per_cluster.pdf"), width = 6, height = 6)
mat <- cell_chat_obj@net$weight
#clusters <- c("cluster0", "cluster15") # clusters of interest
clusters <- as.character(levels(cell_chat_obj@idents))
for (clu in clusters) {
    mat2 <- matrix(0, nrow = nrow(mat), ncol = ncol(mat), dimnames = dimnames(mat))
    mat2[clu, ] <- mat[clu, ]
    netVisual_circle(mat2, vertex.weight = groupSize, weight.scale = TRUE, 
                     vertex.label.cex = 0.55, edge.weight.max = max(mat), 
                     title.name = clu)
}
dev.off()


# Visualize each signaling pathway ----------------------------------------
cell_Net <- subsetCommunication(cell_chat_obj)
table(cell_Net$pathway_name)
pathways_vec <- unique(cell_Net$pathway_name)

# Hierarchy plot
pdf(file.path(out_dir, "3.Hierarchy_plots.pdf"), width = 6, height = 6)
clusters <- as.character(levels(cell_chat_obj@idents))

for (pw in pathways_vec) {
    netVisual_aggregate(cell_chat_obj, signaling = pw,  
                        vertex.receiver = seq_along(clusters))
}
dev.off()

# Circle plot
pdf(file.path(out_dir, "4.Circle_plots.pdf"), width = 6, height = 6)
for (pw in pathways_vec) {
    netVisual_aggregate(cell_chat_obj, signaling = pw, 
                        layout = "circle")
}
dev.off()

# Chord diagram
pdf(file.path(out_dir, "5.Chord_diagrams.pdf"), width = 6, height = 6)
for (pw in pathways_vec) {
    netVisual_aggregate(cell_chat_obj, signaling = pw, 
                        layout = "chord")
}
dev.off()
# pathways_vec <- c('Cd34',"SELL",'PARs','GRN')
# 
# netVisual_heatmap(cell_chat_obj, signaling = c('SELL'))
# 
# # Heatmap
# pdf(file.path(out_dir, "6.Heatmaps.pdf"), width = 6, height = 8)
# for (pw in pathways_vec) {
#   netVisual_heatmap(cell_chat_obj, signaling = pw, color.heatmap = "Red")
# }
# dev.off()

pathways_vec[[4
              ]]

pdf(file.path(out_dir, "8.LR_contributions.pdf"), width = 8, height = 4)
netAnalysis_contribution(cell_chat_obj, signaling = c('CD34','SELL')) + 
    ggtitle("All pathways L-R pairs")
dev.off()

netAnalysis_dot(cell_chat_obj, pattern = "incoming")

pdf(file.path(out_dir, "25.Cell_Communication_patterns_outgoing_river_dot.pdf"), width = 10, height = 15)
# river plot
netAnalysis_river(cell_chat_obj, pattern = "incoming")
# dot plot
netAnalysis_dot(cell_chat_obj, pattern = "incoming")
dev.off()

# Visualize a single ligant-receptor pair per pathway ---------------------
pdf(file.path(out_dir, "9.LR_pairs_per_pathway_hierarchy.pdf"), width = 6, height = 6)
for (pw in pathways_vec){
    pairLR <- extractEnrichedLR(cell_chat_obj, signaling = pw,
                                geneLR.return = FALSE)
    for (idx in seq_len(nrow(pairLR))){
        LR.show <- pairLR[idx, ] # show one ligand-receptor pair
        # Hierarchy plot
        netVisual_individual(cell_chat_obj, signaling = pw, 
                             pairLR.use = LR.show, 
                             vertex.receiver = seq_along(clusters))
    }
}
dev.off()

pdf(file.path(out_dir, "10.LR_pairs_per_pathway_circle.pdf"), width = 6, height = 6)
for (pw in pathways_vec){
    pairLR <- extractEnrichedLR(cell_chat_obj, signaling = pw,
                                geneLR.return = FALSE)
    for (idx in seq_len(nrow(pairLR))){
        LR.show <- pairLR[idx, ] # show one ligand-receptor pair
        # Hierarchy plot
        netVisual_individual(cell_chat_obj, signaling = pw, 
                             pairLR.use = LR.show, 
                             layout = "circle")
    }
}
dev.off()

# Bubble plots
pdf(file.path(out_dir, "12.bubble_plots.pdf"), width = 7, height = 7)
# show all the significant interactions (L-R pairs) from some cell groups (defined by 'sources.use') to other cell groups (defined by 'targets.use')
clusters <- c("Undifferentiated","Meg","Monocyte","Neutrophil")
for (idx in seq_along(clusters) ){
    plot(netVisual_bubble(cell_chat_obj, sources.use = idx, 
                           targets.use = setdiff(seq_along(clusters), idx),
                           remove.isolate = FALSE,
                           title.name = clusters[idx]))
    
    # show all the significant interactions (L-R pairs) associated with the relevant signaling pathways
    relevant.pathways <- unique(cell_Net[(cell_Net$source == clusters[idx]), "pathway_name"])
    
    for (pw in relevant.pathways){
        try(plot(netVisual_bubble(cell_chat_obj, sources.use = idx, 
                                  targets.use = setdiff(seq_along(clusters), idx), 
                                  signaling = pw,
                                  remove.isolate = FALSE,
                                  title.name = paste0(clusters[idx], ": ", pw))))
        # show all the significant interactions (L-R pairs) based on user's input (defined by `pairLR.use`)
        pairLR.use <- extractEnrichedLR(cell_chat_obj, signaling = pw)
        try(print(netVisual_bubble(cell_chat_obj, sources.use = idx, 
                                   targets.use = setdiff(seq_along(clusters), idx),
                                   pairLR.use = pairLR.use, remove.isolate = TRUE,
                                   title.name = paste0(clusters[idx], ": ", "L-R pairs in ", pw))))
    }
}
dev.off()

