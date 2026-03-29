DATASET_PATH=$1
EXP_PATH=$2

python active_train.py -s $DATASET_PATH -m ${EXP_PATH} --eval --method=warprf --seed=0 --schema v20seq1_inplace --iterations 30000
python render.py -m ${EXP_PATH} --skip_train
python metrics.py -m ${EXP_PATH}

    