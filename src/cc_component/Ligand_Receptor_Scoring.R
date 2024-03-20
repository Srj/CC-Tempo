# library(Matrix)
# library(Seurat)
# library(CellChat)
# library(reticulate)
# library(data.table)
# library(ggalluvial)

# seurat_obj <- readRDS("data/weinreb_seurat_scaled.rds")

ligand_receptor_scoring <- function(seurat_obj,fname){

  #Receptor Scoring
  APP <- c('Cd74')
  CADM <- c("Cadm1", 'Cadm3')
  CCL <- c('Ccr2', 'Ccr1l1', 'Ccr4', 'Ccr1', 'Ccr3', 'Ackr1', 'Ccr5', 'Ccr9')
  CD34 <- c('Selp')
  CD45 <- c('Mrc1', 'Cd22')
  CD86 <- c('Cd28')
  CDH <- c('Cdh15', 'Cdh1', 'Cdh2', 'Cdh4')
  COMPLEMENT <- c('Itgax', 'C3ar1', 'Itgam','Itgb2', 'Cr2', 'C5ar1')
  CSF <- c('Csf1r')
  CXCL <- c('Cxcr2', 'Cxcr1', 'Cxcr3', 'Cxcr5', 'Ackr1', 'Cxcr6')
  GALECTIN <- c('Cd44', 'Ptprc', 'Havcr2', 'Ighm')
  GRN <- c('Sort1')
  ICAM <- c('Itgal', 'Itgam', 'Itgax', 'Itgb2', 'Itgb2l', 'Spn')
  IL6 <- c('Il11ra1', 'Il6r','Il6st', 'Il31ra','Osmr')
  ITGAL_ITGB2 <- c('Icam1', 'Icam2', 'Cd226')
  JAM <- c('F11r', 'Itgam', 'Itgb2', 'Itgal', 'Jam2', 'Itgb2l', 'Jam3', 'Itga3', 'Itgb1', 'Itgav')
  LAMININ <- c('Cd44','Dag1','Itga1','Itga2','Itga3','Itga6','Itga7','Itga9','Itgav','Itgb1','Itgb8','Sv2a','Sv2b','Sv2c')
  MIF <- c('Cd74', 'Cd44','Cxcr2', 'Cxcr4')
  PARs <- c('F2r', 'F2rl3', 'F2rl2')
  PECAM1 <- c('Pecam1')
  PROS <- c("Axl", "Tyro3")
  SELL <- c('Cd34', 'Podxl')
  SELPLG <- c('Selp', 'Sell')
  SEMA4 <- c('Nrp1','Plxna2', 'Plxna3', 'Plxnb2', 'Cd72', 'Plxna1', 'Plxnb3', 'Timd2', 'Plxna4')
  SEMA7 <- c('Itgb1','Itga1', 'Plxnc1')
  SPP1 <- c('Cd44','Itga4','Itga5','Itga8','Itga9','Itgav','Itgb1','Itgb3','Itgb5','Itgb6')
  THBS <- c('Cd36', 'Cd47', 'Sdc1', 'Itgav','Itgb3', 'Sdc4', 'Itga3','Itgb1')
  THY1 <- c("Itgam", "Itgb2", "Itgax", "Itgav", "Itgb3")
  VWF <- c('Itgav','Itgb3','Itga2b','Itgb3','Gp1ba','Gp1bb','Gp5','Gp9')

  receptors <- list(APP, CADM,
    CCL, CD34, CD45, CD86, CDH, COMPLEMENT, CSF, CXCL, GALECTIN, GRN, ICAM, IL6, ITGAL_ITGB2,
    JAM, LAMININ, MIF, PARs, PECAM1, PROS, SELL, SELPLG, SEMA4, SEMA7, SPP1, THBS, THY1, VWF)

  seurat_obj <- AddModuleScore(seurat_obj, features = receptors, name = "receptor", slot = "data")

  receptor_scores <- seurat_obj[[]][,c('receptor1',
                                     'receptor2','receptor3','receptor4','receptor5','receptor6','receptor7','receptor8',
                                     'receptor9','receptor10','receptor11','receptor12','receptor13','receptor14','receptor15',
                                     'receptor16','receptor17','receptor18','receptor19','receptor20','receptor21','receptor22',
                                     'receptor23','receptor24','receptor25','receptor26','receptor27','receptor28','receptor29')]

  colnames(receptor_scores) <- c("APP","CADM","CCL", "CD34", "CD45", "CD86", "CDH", "COMPLEMENT", "CSF", "CXCL",
                               "GALECTIN", "GRN", "ICAM", "IL6", "ITGAL_ITGB2", "JAM", "LAMININ",
                               "MIF", "PARs", "PECAM1", "PROS", "SELL", "SELPLG", "SEMA4", "SEMA7",
                               "SPP1", "THBS", "THY1", "VWF")
  write.csv(receptor_scores, file.path("notebook","fold_change_score",paste0("receptors_scoring_29_,",fname,".csv")), row.names = F)
  
  #Ligand Scoring
  APP <- c('App')
  CADM <- c("Cadm1", 'Cadm3')
  CCL <- c('Ccl1','Ccl12','Ccl17','Ccl2','Ccl22','Ccl24','Ccl25','Ccl3','Ccl4','Ccl5','Ccl6','Ccl7','Ccl8','Ccl9')
  CD34 <- c('Cd34')
  CD45 <- c('Ptprc')
  CD86 <- c('Cd86')
  CDH <- c('Cdh15', 'Cdh1', 'Cdh2', 'Cdh4')
  COMPLEMENT <- c('C4a', 'Hc', 'C3')
  CSF <- c('Csf1', 'Il34')
  CXCL<- c('Cxcl1','Cxcl10','Cxcl11','Cxcl13','Cxcl16','Cxcl2','Cxcl5','Cxcl9','Pf4','Ppbp')
  GALECTIN <- c('Lgals9')
  GRN <- c('Grn')
  ICAM <- c('Icam1', 'Icam2')
  IL6 <- c('Il6', 'Il11', 'Il31')
  ITGAL_ITGB2 <- c('Itgal','Itgb2', 'Itgb2l')
  JAM <- c('Jam2', 'Jam3', 'F11r')
  LAMININ <- c('Lama5', 'Lamb2', 'Lamb3', 'Lamc1', 'Lamc2')
  MIF <- c('Mif')
  PARs <- c('Prss2', 'Ctsg', 'Gzma')
  PECAM1 <- c('Pecam1')
  PROS <- c("Pros1")
  SELL <- c('Sell')
  SELPLG <- c('Selplg')
  SEMA4 <- c('Sema4a', 'Sema4c', 'Sema4d')
  SEMA7 <- c('Sema7a')
  SPP1 <- c('Spp1')
  THBS <- c('Thbs3', 'Thbs2', 'Thbs1', 'Thbs4')
  THY1 <- c("Thy1")
  VWF <- c('Vwf')


  ligands <- list(APP,CADM,CCL,CD34,CD45,CD86,CDH,COMPLEMENT,CSF,CXCL,GALECTIN,GRN,ICAM,IL6,ITGAL_ITGB2,
                  JAM,LAMININ,MIF,PARs,PECAM1,PROS,SELL,SELPLG,SEMA4,SEMA7,SPP1,THBS,THY1,VWF)
  
  seurat_obj <- AddModuleScore(seurat_obj, features = ligands, name="ligand", slot = "data")
  
  ligand_scores <- seurat_obj[[]][,c('ligand1','ligand2','ligand3','ligand4','ligand5','ligand6','ligand7','ligand8','ligand9',
                                     'ligand10','ligand11','ligand12','ligand13','ligand14','ligand15','ligand16','ligand17',
                                     'ligand18','ligand19','ligand20','ligand21','ligand22','ligand23','ligand24',
                                     'ligand25','ligand26','ligand27','ligand28','ligand29')]
  colnames(ligand_scores) <- c("APP","CADM","CCL","CD34","CD45","CD86","CDH","COMPLEMENT",
                               "CSF","CXCL","GALECTIN","GRN","ICAM","IL6","ITGAL_ITGB2","JAM",
                               "LAMININ","MIF","PARs","PECAM1","PROS","SELL","SELPLG","SEMA4","SEMA7",
                               "SPP1","THBS","THY1","VWF")
  write.csv(ligand_scores, file.path("notebook","fold_change_score",paste0("ligands_scoring_29_,",fname,".csv")), row.names = F)

}
