DATASET_PATH=$1
EXP_PATH=$2

python active_train.py -s $DATASET_PATH -m ${EXP_PATH} --eval --method=warprf --seed=0 --schema v10seq1_inplace --iterations 30000 --white_background
python render.py -m ${EXP_PATH} --skip_train
python metrics.py -m ${EXP_PATH}
