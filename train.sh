DATA_PATH='/home/spyder/hazel/pointnet_geoerror/dataset/'
LABEL_PATH='/home/spyder/hazel/dataset_4axisMT/model4_improved_10class/model4_alldata_10_label.csv'
#LABEL_PATH='/home/spyder/hazel/dataset_4axisMT/alldata_29/alldata_label.csv'
#NUMPY_PATH='/home/spyder/hazel/dataset_4axisMT/alldata_29/alldata.csv'
NUMPY_PATH="/home/spyder/hazel/dataset_4axisMT/model4_improved_10class/model4_alldata_10.csv"
MODE='train'
LR=1e-5
BATCHSIZE=8
EPOCH=60
PRETRAINED=None
FREQ=1
opt="Adam"
outf='output/model_8_1e5_0.3_20_0.4'
DROP_PROB=0.3
NUM_CLS=10
STEP_SIZE=20
STEP_GAMMA=0.4
MODEL=pointnet

# 可以修改的参数：DROP_PROB LR BATCH_SIZE
# STEP_SIZE 修改间隔epoch数之后 学习率调整为之前的STEP_GAMMA倍

python train_classification_geoerror.py --data_path $DATA_PATH --nepoch $EPOCH  --batchSize $BATCHSIZE \
        --label_path $LABEL_PATH --numpy_path $NUMPY_PATH --lr $LR \
        --mode $MODE --pretrained $PRETRAINED --validate_freq $FREQ --outf $outf --opt $opt --cls_num $NUM_CLS \
        --drop_prob $DROP_PROB --step_size $STEP_SIZE --step_gamma $STEP_GAMMA --model $MODEL
