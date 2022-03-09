from model.pointnet import PointNetClsGeoerror
from model.pointnet_2 import PointNet2ClsGeoerror

def get_model(opt):
    model_name = opt.model
    if model_name == 'pointnet':
        model = PointNetClsGeoerror(feature_transform=False, opt=opt)
    elif model_name == 'pointnet2':
        model = PointNet2ClsGeoerror(opt)
    else:
        raise NotImplemented('wrong model name')
    return model
