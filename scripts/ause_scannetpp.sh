DATASET_PATH=$1
EXP_PATH=$2


python active_train.py -s $DATASET_PATH -m $EXP_PATH --schema=all --eval --iterations 2000
python evaluate_uncertainty.py -m $EXP_PATH --gtdepth scannetpp --viz