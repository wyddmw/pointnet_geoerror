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