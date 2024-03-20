library(Matrix)
library(Seurat)
library(CellChat)
library(reticulate)
library(data.table)
library(ggalluvial)
library(NMF)
library(gridExtra)
set.seed(1)
gene.shuffle <- T

seurat_obj <- readRDS("data/weinreb_seurat_scaled.rds")
reduces_seurat_obj <- subset(seurat_obj, subset=(Time.point == 2))
#Extract Expression & Metadata
exprMat <- GetAssayData(seurat_obj, assay = "RNA", slot = "data")
meta.data <- seurat_obj@meta.data

np <- import("numpy")
day2prog <- np$load("data/ProgCellChatIndex.npy", allow_pickle = T)
cc <- c(day2prog)

cell_chat_obj <- createCellChat(object = exprMat[, cc], meta = meta.data[cc, ], group.by = "Cell.type.annotation")
cell_chat_obj@DB <- CellChatDB.mouse
cell_chat_obj <- subsetData(cell_chat_obj)
cell_chat_obj <- identifyOverExpressedGenes(cell_chat_obj) %>%
  identifyOverExpressedInteractions()
cell_chat_obj <- projectData(cell_chat_obj, PPI.mouse)
cell_chat_obj <- computeCommunProb(cell_chat_obj) %>%
  filterCommunication(min.cells = 10)
cell_chat_obj <- aggregateNet(cell_chat_obj)

# Compute the communication probability on signaling pathway level
cell_chat_obj <- computeCommunProbPathway(cell_chat_obj)

np <- import("numpy")
np$save("Mono_Neutro_Prog_Prob", cell_chat_obj@netP$prob)
np$save("Mono_Neutro_Prog_Pathways", cell_chat_obj@netP$pathways)

saveRDS(cell_chat_obj,"data/cell_chat_obj_Day2_Progenitors.rds")
cell_chat_obj <- readRDS("data/cell_chat_obj_Day2.rds")


cell_chat_obj <- aggregateNet(cell_chat_obj)

cell_chat_obj <- identifyCommunicationPatterns(cell_chat_obj, 
                                               pattern = "incoming", 
                                               k = 5, # per the above plot
                                               width = 10, height = 15)
cell_chat_obj <- identifyCommunicationPatterns(cell_chat_obj, 
                                               pattern = "outgoing", 
                                               k = 5, # per the above plot
                                               width = 10, height = 15)

pdf("Day2_Dot_Plot.pdf", width = 6, height = 6)
netAnalysis_dot(cell_chat_obj, pattern = "incoming", slot.name = "netP")
netAnalysis_dot(cell_chat_obj, pattern = "outgoing", slot.name = "netP")
dev.off()


pdf("Day_Chord_Diagram.pdf", width = 6, height = 6)
netVisual_chord_gene(cell_chat_obj, sources.use = c(1:11), targets.use = c(1:11), slot.name = "netP", legend.pos.x = 10)
# netVisual_chord_gene(cell_chat_obj, sources.use = c(1:11), targets.use = c(11), slot.name = "netP", legend.pos.x = 10)
dev.off()


# Chord diagram
cell_chat_obj <- readRDS("data/cell_chat_obj_Day2.rds")
cell_Net <- subsetCommunication(cell_chat_obj)
table(cell_Net$pathway_name)
pathways_vec <- unique(cell_Net$pathway_name)

pdf(file.path("Day2_Chord_diagrams.pdf"), width = 6, height = 6)
for (pw in pathways_vec) {
  netVisual_aggregate(cell_chat_obj, signaling = pw, 
                      layout = "chord")
}
dev.off()


# randomly shuffle the index(genes)
rand_idx <- sample(nrow(exprMat))

# Days in the Dataset
DAYS = c(2,4,6)


# create cellchat object for each day t in DAYS
for(t in DAYS){
  print(t)
  
  # subset the seurat object for day t
  reduced_seurat_obj <- subset(seurat_obj, (Time.point == t))
  
  #Extract Expression & Metadata
  exprMat <- GetAssayData(reduced_seurat_obj, assay = "RNA", slot = "data")
  meta.data <- reduced_seurat_obj@meta.data
  
  #Shuffle the exprMat
  if(gene.shuffle == T){
    print("Shuffling GEX Data")
    row.names <- rownames(exprMat)
    exprMat <- exprMat[rand_idx,]
    rownames(exprMat) <- row.names
  }
  
  #Create CellChat Object & run the CellChat Pipeline
  cell_chat_obj <- createCellChat(object = exprMat, meta = meta.data, group.by = "Cell.type.annotation")
  cell_chat_obj@DB <- CellChatDB.mouse
  cell_chat_obj <- subsetData(cell_chat_obj)
  cell_chat_obj <- identifyOverExpressedGenes(cell_chat_obj) %>%
    identifyOverExpressedInteractions()
  cell_chat_obj <- projectData(cell_chat_obj, PPI.mouse)
  cell_chat_obj <- computeCommunProb(cell_chat_obj) %>%
    filterCommunication(min.cells = 10)
  cell_chat_obj <- aggregateNet(cell_chat_obj)
  
  # Compute the communication probability on signaling pathway level
  cell_chat_obj <- computeCommunProbPathway(cell_chat_obj)
  cell_chat_obj <- aggregateNet(cell_chat_obj)
  cell_chat_obj <- identifyCommunicationPatterns(cell_chat_obj, 
                                                 pattern = "incoming", 
                                                 k = 8, # per the above plot
                                                 width = 10, height = 15)
  cell_chat_obj <- identifyCommunicationPatterns(cell_chat_obj, 
                                                 pattern = "outgoing", 
                                                 k = 8, # per the above plot
                                                 width = 10, height = 15)
  
  # save the cellchat object for day t
  saveRDS(cell_chat_obj, file = paste0("data/cell_chat_obj_Day", t, ".rds"))
}



for(t in c(2,4,6)){
  cell_chat_obj <- readRDS(paste0("data/cell_chat_obj_Day", t, ".rds"))
  
  pdf(paste0("data/Incoming_Signal_Day",t,".pdf"), width = 10, height = 6)
  print(netAnalysis_dot(cell_chat_obj, pattern = "incoming", slot.name = "netP"))
  print(netAnalysis_dot(cell_chat_obj, pattern = "outgoing", slot.name = "netP"))
  dev.off()
  
  #Extracting Incoming Matrix
  for (pattern in c("incoming", "outgoing")){
  
    patternSignaling <- cell_chat_obj@netP$pattern[[pattern]]
    data1 <- patternSignaling$pattern$cell
    data2 <- patternSignaling$pattern$signaling
    data <- patternSignaling$data
    
    cutoff <- 1/length(unique(data1$Pattern))
    
    data1$Contribution[data1$Contribution < cutoff] <- 0
    data2$Contribution[data2$Contribution < cutoff] <- 0
    
    data3 <- merge(data1, data2, by.x = "Pattern", by.y = "Pattern")
    data3$Contribution <- data3$Contribution.x * data3$Contribution.y
    data3 <- data3[,colnames(data3) %in% c("CellGroup","Signaling","Contribution")]
    
    data <- as.data.frame(as.table(data))
    data <- data[data[,3] != 0,]
    data12 <- paste0(data[,1],data[,2])
    data312 <- paste0(data3[,1], data3[,2])
    
    idx1 <- which(match(data312,data12, nomatch = 0) == 0)
    data3$Contribution[idx1] <- 0
    data3$id <- data312
    data3 <- data3 %>% group_by(id) %>% top_n(1, Contribution)
    data3$Contribution[which(data3$Contribution == 0)] <- NA
    write.csv(data3,paste0(pattern,"_comm_weinreb", t,".csv"), row.names = F)
  
  }
}










t <- 2
cell_chat_obj <-readRDS(paste0("data/Neutro_Mono_All_Day.rds"))
cell_chat_obj <- computeCommunProbPathway(cell_chat_obj)
unique(cell_chat_obj@idents)
np <- import("numpy")
np$save("Mono_Neutro_Prob", cell_chat_obj@netP$prob)
np$save("Mono_Neutro_Pathways", cell_chat_obj@netP$pathways)


#Pathway List Determine
pairLR.use <- cell_chat_obj@LR$LRsig
prob <- cell_chat_obj@net$prob
net <- cell_chat_obj@net
prob[net$pval > 0.05] <- 0
pathways <- unique(pairLR.use$pathway_name)
group <- factor(pairLR.use$pathway_name, level = pathways)
prob.pathways <- aperm(apply(prob, c(1,2),by,group,sum),c(2,3,1))

pathways

write.csv(pairLR.use, "Signalling_pathways.csv", row.names = F)





# visualize interactions --------------------------------------------------
pdf(file.path("1.cell_cell_communications.pdf"), width = 6, height = 6)
groupSize <- as.numeric(table(cell_chat_obj@idents))
netVisual_circle(cell_chat_obj@net$count, vertex.weight = groupSize, vertex.label.cex = 0.75,
                 weight.scale = TRUE, label.edge = FALSE, title.name = "Number of interactions") 
netVisual_circle(cell_chat_obj@net$weight, vertex.weight = groupSize, vertex.label.cex = 0.75,
                 weight.scale = TRUE, label.edge = FALSE, title.name = "Interaction weights/strength")
dev.off()








np <- import("numpy")
prob <- np$load("prob.npy")
pval <- np$load("pval.npy")

thresh <- 0.005
pval[prob == 0] <- 1
prob[pval >= thresh] <- 0
count <- apply(prob > 0, c(1,2), sum)
weight <- apply(prob, c(1,2), sum)
weight[is.na(weight)] <- 0
count[is.na(count)] <- 0
pdf(file.path("1.cell_cell_communications_PY.pdf"), width = 6, height = 6)
netVisual_circle(count, vertex.weight = groupSize, vertex.label.cex = 0.75,
                 weight.scale = TRUE, label.edge = FALSE, title.name = "Number of interactions") 
netVisual_circle(weight, vertex.weight = groupSize, vertex.label.cex = 0.75,
                 weight.scale = TRUE, label.edge = FALSE, title.name = "Interaction weights/strength")
dev.off()



pdf(file.path("2.communications_per_cluster_PY_9.pdf"), width = 6, height = 6)

mat <- count
#clusters <- c("cluster0", "cluster15") # clusters of interest
clusters <- as.character(levels(cell_chat_obj@idents))

mat2 <- matrix(0, nrow = nrow(mat), ncol = ncol(mat), dimnames = dimnames(mat))
mat2[9, ] <- mat[9, ]
netVisual_circle(mat2, vertex.weight = groupSize, weight.scale = TRUE, 
                   vertex.label.cex = 0.55, edge.weight.max = max(mat), 
                   title.name = clu)

dev.off()



dimnames(mat) <- list(1,11)

clu

pdf(file.path("2.communications_per_cluster.pdf"), width = 6, height = 6)

mat <- cell_chat_obj@net$count
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


pdf(file.path("2.communications_per_cluster.pdf"), width = 6, height = 6)

mat <- cell_chat_obj@net$count
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

pdf(file.path("1.cell_cell_communications.pdf"), width = 6, height = 6)
groupSize <- as.numeric(table(cell_chat_obj@idents))
netVisual_circle(cell_chat_obj@net$count, vertex.weight = groupSize, vertex.label.cex = 0.75,
                 weight.scale = TRUE, label.edge = FALSE, title.name = "Number of interactions") 
netVisual_circle(cell_chat_obj@net$weight, vertex.weight = groupSize, vertex.label.cex = 0.75,
                 weight.scale = TRUE, label.edge = FALSE, title.name = "Interaction weights/strength")
dev.off()

# cell_chat_obj <- readRDS("cell_chat_obj.RDS")
net <- cell_chat_obj@net$prob
# 
# # Save for Numpy
np = import("numpy")
np$save("cellchat_score.npy", r_to_py(net))
fwrite(dimnames(net)[1], file = "cluster_order.txt")

fwrite(dimnames(net)[3], file ="ligand_receptors.txt")
