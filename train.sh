DATA_PATH='/home/spyder/hazel/dataload-V2/origin/'
LABEL_PATH='/home/spyder/hazel/dataload-V2/label13.csv'
NUMPY_PATH='/home/spyder/hazel/dataload-V2/all_data.txt'
MODE='train'
LR=1e-3
BATCHSIZE=32
EPOCH=30
#MODEL='cls/cls_model_0.pth'
MODEL=None
FREQ=10

python train_classification_geoerror.py --data_path $DATA_PATH --nepoch $EPOCH  --batchSize $BATCHSIZE --label_path $LABEL_PATH --numpy_path $NUMPY_PATH --lr $LR --mode $MODE --model $MODEL --validate_freq $FREQ
