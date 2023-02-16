import torch
from torch.autograd import Variable
import argparse
import random
import os
import models
import dataLoader_clearpose
from torch.utils.data import DataLoader
import os.path as osp
from model.pointnet import PointNetRefinePoint
from chamfer_distance import ChamferDistance
import open3d as o3d

parser = argparse.ArgumentParser()

parser.add_argument('--dataRoot', default='', help='path to images' )
parser.add_argument('--experiment', default=None, help='the path to store samples and models' )
parser.add_argument('--outputRoot', default=None, help='the path to output the code')
# The basic training setting
parser.add_argument('--nepoch', type=int, default=10, help='the number of epochs for training' )
parser.add_argument('--imageHeight', type=int, default=480, help='the height / width of the input image to network' )
parser.add_argument('--imageWidth', type=int, default=640, help='the height / width of the input image to network' )
parser.add_argument('--envHeight', type=int, default=256, help='the height / width of the input envmap to network' )
parser.add_argument('--envWidth', type=int, default=512, help='the height / width of the input envmap to network' )
# Weight of Loss
parser.add_argument('--normalWeight', type=float, default=5.0, help='the weight of normal' )
parser.add_argument('--pointWeight', type=float, default=200.0, help='the weight of point' )
# The gpu setting
parser.add_argument('--cuda', action='store_true', help='enables cuda' )
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for training network' )
# The view selection mode
parser.add_argument('--viewMode', type=int, default=0, help='the view selection Mode: 0-renderError, 1-nearest, 2-average')
# The loss function
parser.add_argument('--lossMode', type=int, default=2, help='the loss function: 0-view, 1-nearest, 3-chamfer')

# clearpose dataset
parser.add_argument('--shapeStart', type=int, default=0, help='the start id of the shape')
parser.add_argument('--shapeEnd', type=int, default=1, help='the end id of the shape')

parser.add_argument('--camNum', type=int, default=5, help='the number of views to create the visual hull' )
parser.add_argument('--meta_file', type=str, default='metadata.mat')
parser.add_argument('--isBaseLine', action='store_true', help='whether to output the baseline with only the normal mapping')
parser.add_argument('--isNoRenderError', action='store_true', help='whether to use rendering error or not')
parser.add_argument('--eta1', type=float, default=1.0003, help='the index of refraction of air' )
parser.add_argument('--eta2', type=float, default=1.4723, help='the index of refraction of glass' )
parser.add_argument('--fov', type=float, default=63.4, help='the field of view of camera' )
parser.add_argument('--env_path', type=str, help='env map path')

opt = parser.parse_args()
print(opt)

# opt.dataRoot = opt.dataRoot % opt.camNum
opt.gpuId = opt.deviceIds[0]
nw = opt.normalWeight
pw = opt.pointWeight

opt.experiment = 'TransparentShapeReconstruction/PointCloud/check5_point_nw5.00_pw200.00_view_renderError_loss_chamfer'

opt.batchSize = opt.camNum

opt.outputRoot = opt.experiment.replace('check', 'output')

os.system('mkdir {0}'.format( opt.outputRoot ) )
os.system('cp *.py %s' % opt.outputRoot )

opt.seed = 0
print("Random Seed: ", opt.seed )
random.seed(opt.seed )
torch.manual_seed(opt.seed )

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda" )

shapeName = osp.join(opt.outputRoot, 'pointCloud_%d_view_' % opt.camNum )
if opt.viewMode == 0:
    shapeName += 'renderError'
elif opt.viewMode == 1:
    shapeName += 'nearest'
elif opt.viewMode == 2:
    shapeName += 'average'

if opt.isNoRenderError:
    shapeName += '_norendering'

if opt.isBaseLine:
    shapeName += '_baseLine.ply'
else:
    shapeName += '_loss_'
    if opt.lossMode == 0:
        shapeName += 'view'
    elif opt.lossMode == 1:
        shapeName += 'nearest'
    elif opt.lossMode == 2:
        shapeName += 'chamfer'

brdfDataset = dataLoader_clearpose.BatchLoader(opt)

brdfLoader = DataLoader( brdfDataset, batch_size = 1, num_workers = 16, shuffle = False )

sampler = models.groundtruthSampler(
        camNum = opt.camNum,
        fov = opt.fov,
        imHeight = opt.imageHeight,
        imWidth = opt.imageWidth,
        isNoRenderError = True )

# Define the model and optimizer
lr_scale = 1
pointNet = PointNetRefinePoint()
pointNet.load_state_dict(torch.load('%s/pointNet_%d.pth' % (opt.experiment, opt.nepoch-1) ) )
chamferDist = ChamferDistance()
if opt.cuda:
    pointNet = pointNet.cuda()
    chamferDist = chamferDist.cuda()

epoch = opt.nepoch
testingLog = open('{0}/testingLog_{1}.txt'.format(opt.outputRoot, epoch ), 'w' )

renderer = models.renderer(eta1 = opt.eta1, eta2 = opt.eta2,
        isCuda = opt.cuda, gpuId = opt.gpuId,
        batchSize = opt.batchSize,
        fov = opt.fov,
        imWidth=opt.imageWidth, imHeight = opt.imageHeight,
        envWidth = opt.envWidth, envHeight = opt.envHeight )


for i, dataBatch in enumerate(brdfLoader ):
    for key in dataBatch:
        dataBatch[key] = dataBatch[key].squeeze(0).cuda()

    error = torch.zeros_like(dataBatch['seg1Int'])


    refraction, reflection, maskTr = renderer.forward(
            dataBatch['origin'], dataBatch['lookat'], dataBatch['up'],
            dataBatch['env'], dataBatch['normal1Opt'], dataBatch['normal2Opt'] )
    
    maskTr = (1 - maskTr) * dataBatch['seg1Int'] # TODO: figure out where to get maskTr, from 'seg1'?

    if opt.viewMode == 0:
        feature, gtNormal, gtPoint, viewIds = sampler.sampleBestView(
                dataBatch['origin'], dataBatch['lookat'], dataBatch['up'],
                dataBatch['pointVH'], dataBatch['normalPointVH'],
                dataBatch['point'], dataBatch['normalPoint'],
                maskTr, dataBatch['normal1Opt'], error, dataBatch['depth1VH'], dataBatch['seg1Int'],
                dataBatch['depth1'], dataBatch['normal1'] )
    elif opt.viewMode == 1:
        feature, gtNormal, gtPoint, viewIds = sampler.sampleNearestView(
                dataBatch['origin'], dataBatch['lookat'], dataBatch['up'],
                dataBatch['pointVH'], dataBatch['normalPointVH'],
                dataBatch['point'], dataBatch['normalPoint'],
                maskTr, dataBatch['normal1Opt'], error, dataBatch['depth1VH'], dataBatch['seg1Int'],
                dataBatch['depth1'], dataBatch['normal1'] )
    elif opt.viewMode == 2:
        feature, viewIds = sampler.sampleNearestViewAverage(
                dataBatch['origin'], dataBatch['lookat'], dataBatch['up'],
                dataBatch['pointVH'], dataBatch['normalPointVH'],
                dataBatch['point'], dataBatch['normalPoint'],
                maskTr, dataBatch['normal1Opt'], error, dataBatch['depth1VH'], dataBatch['seg1Int'],
                dataBatch['depth1'], dataBatch['normal1'] )

    normalInitial = feature[:, 0:3].clone()

    pointPredDelta, normalPred = pointNet(
            dataBatch['pointVH'].unsqueeze(0).permute([0, 2, 1]),
            normalInitial.unsqueeze(0).permute([0, 2, 1]),
            feature.unsqueeze(0).permute([0, 2, 1])  )
    
    pointPredDelta = pointPredDelta.squeeze(0 )
    normalPred = normalPred.squeeze(0 )

    pointPred = dataBatch['pointVH'] + pointPredDelta
    pointArr = pointPred.data.cpu().numpy()
    normalArr = normalPred.data.cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointArr )
    pcd.normals = o3d.utility.Vector3dVector(normalArr )
    pcd.colors = o3d.utility.Vector3dVector(0.5 * (normalArr + 1) )
    o3d.io.write_point_cloud(shapeName + f'_{i}.ply', pcd)

testingLog.close()