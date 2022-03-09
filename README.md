# PointNet For Geoerror Classification
This is a simplified model based on the PointNet and PointNet++ for geometry error classification.

## Environment
1. PyTorch 1.5
2. Numpy
3. Matplotlib

## Training
Specify the parameter MODEL in train.sh as pointnet2 for PointNet++ and pointnet for PointNet training.
```
sh train.sh
```
The training data is not uploaded. Follow the next command to test the model with fake random data instead.

## Quick Test
```
# Check PointNet
python model/pointnet.py
# Check PointNet++
python model/pointnet_2.py
```
## Guidance
First we need to initialize the training environment:
```
1. source env.sh
```
Then we start training with the specific training parameters in scripts/train.sh for each training
```
MODEL_TAG       # model tag for visualization
CLASSIFIER      # softmax or sigmoid
MODEL           # pointnet, pointnet2, bp
AC_FN           # sigmoid, relu
outf            # output dir
```
Start training with the following command:
```
sh scripts/train.sh
```
For visualizing the accuracy on both testing and training set on specific model, modify the following parameters in utils/ac_vis.py
```
accuracy_list
output_dir
# run visualization
python utils/ac_vis.py
```

For evaluation, specify the following parameters in the scripts/train.sh
```
PRETRAINED
MODE='testing'
BATCHSIZE=1
```