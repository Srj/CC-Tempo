# pip install umap-learn geomloss shap

GENE_FILE=/content/drive/MyDrive/SrJ/CC-Tempo/data/full_dataset_normalized_scaled.csv  #Path with Gene and CellChat Score
CELL_FILE=/content/drive/MyDrive/SrJ/CC-Tempo/data/cc_score_random.csv
METADATA_FILE=/content/drive/MyDrive/SrJ/CC-Tempo/data/full_dataset_metadata.csv
WEIGHT_FILE=/content/drive/MyDrive/SrJ/CC-Tempo/data/Weinreb2020_growth-all_kegg.pt
DATA_DIR=/content/drive/MyDrive/SrJ/CC-Tempo/data/Prescient_CC_Comp_Added_CC_Not_Scaled  #Path to Save data.pt  
OUT_DIR=/content/drive/MyDrive/SrJ/CC-Tempo/models/Prescient_CC_Comp_Added_CC_Not_Scaled_Loss_Combined_No_Marginal #Directory Containing My Model Config & model weights

# Copy if data.pt not exists
# cp -r -n $DATA_DIR/processed_data/ .
# DATA_DIR=.
# echo "Processing Data ..."

# Write Log
# rm -rIn $DATA_DIR
# mkdir $DATA_DIR
# cat process_data.py > $DATA_DIR/code.log

# python process_data.py -d $GENE_FILE -m $METADATA_FILE --growth_path $WEIGHT_FILE\
#   -o $DATA_DIR --tp_col 'Time point' --celltype_col 'Annotation' --num_pcs 50 -c $CELL_FILE

rm -rI  $OUT_DIR
mkdir $OUT_DIR

cat train.py > $OUT_DIR/code.log 
cat run.py >> $OUT_DIR/code.log 
cat models.py >> $OUT_DIR/code.log 
python train.py --gpu 0 -i $DATA_DIR/data.pt --out_dir $OUT_DIR --weight_name 'kegg-growth'


