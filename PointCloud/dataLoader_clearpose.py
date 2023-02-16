import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from scipy.io import loadmat
import cv2
import os.path as osp
import numpy as np
import os
import h5py


class BatchLoader(Dataset):
    def __init__(self, opt):

        self.dataRoot = opt.dataRoot
        self.imHeight = opt.imageHeight
        self.imWidth = opt.imageWidth
        self.phase = 'TEST'
        self.frame_interval = 200
        self.opt = opt

        self.shape_idx = list(range(opt.shapeStart, opt.shapeEnd))

    def load_camera(self):
        meta_filepath = f'{self.data_folder}/metadata.mat'
        metadata = loadmat(meta_filepath)
        cam_file = [f for f in os.listdir(self.data_folder) if f.endswith('txt')][0]
        self.camNum = int(cam_file.replace('cam', '').replace('.txt', ''))
        camDict = {}
        for i in range(0, self.camNum*self.frame_interval+1, self.frame_interval): # we hardcode frame sample rate as 200
            paramDict = {} # copied from TransparentShapeRealData/createRealData/3_computeVisualHull.py
            rt = metadata[f'{i:06d}']['rotation_translation_matrix'][0][0]
            R, T = rt[:, :3], rt[:, 3]
            R, T = R.T, -R.T @ T
            paramDict['Rot'] = R
            paramDict['Trans'] = T
            paramDict['Origin'] = -R.T @ T
            paramDict['Target'] = R[2] + paramDict['Origin']
            paramDict['Up'] = -R[1]
            paramDict['cId'] = i
            paramDict['imgName'] = f'{self.dataRoot}/{i:06d}-color.png'
            camDict[i] = paramDict
        return camDict

    def __len__(self):
        return len(self.shape_idx)

    def __getitem__(self, ind):
        # normalize the normal vector so that it will be unit length
        self.data_folder = f'{self.dataRoot}/Shape__{ind:d}/'
        camDict = self.load_camera()
        
        imNames = []
        seg1Ints = []
        normal1s = []
        normal2s = []
        depth1s = []
        depth1VHs = []
        envs = []

        origins = []
        lookats = []
        ups = []

        for ind in range(self.camNum):
            cam_ind = ind*self.frame_interval
            imName = camDict[cam_ind]['imgName']
            origin, lookat, up = camDict[cam_ind]['Origin'], camDict[cam_ind]['Target'], camDict[cam_ind]['Up'] # check target == lookat
            
            seg = cv2.imread(f'{self.data_folder}/seg_{ind+1:d}.png', -1)
            seg[seg != 0] = 1
            hf = h5py.File(f'{self.data_folder}/imVH_twoBounce_{ind+1}.h5', 'r')
            twoBounce = np.array(hf.get('data'), dtype=np.float32 )
            hf.close()

            if twoBounce.shape[0] != self.imWidth or twoBounce.shape[1] != self.imHeight:
                newTwoBounce1 = cv2.resize(twoBounce[:, :, 0:3], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                newTwoBounce2 = cv2.resize(twoBounce[:, :, 3:6], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )

                newTwoBounce4 = cv2.resize(twoBounce[:, :, 6:9], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )
                newTwoBounce5 = cv2.resize(twoBounce[:, :, 9:12], (self.imWidth, self.imHeight ), interpolation=cv2.INTER_AREA )

                twoBounce = np.concatenate((newTwoBounce1, newTwoBounce2, newTwoBounce4, newTwoBounce5), axis=2)

            normal1 = twoBounce[:, :, 0:3].transpose([2, 0, 1] )
            normal1 = np.ascontiguousarray(normal1 )
            # TODO: normal = cv2.imread(PATH) / 255
            normal1 = normal1 / np.sqrt(np.maximum(np.sum(normal1 * normal1, axis=0), 1e-10) )[np.newaxis, :]
            normal1 = normal1 * seg


            normal2 = twoBounce[:, :, 6:9].transpose([2, 0, 1] )
            normal2 = np.ascontiguousarray(normal2 )
            normal2 = normal2 / np.sqrt(np.maximum(np.sum(normal2 * normal2, axis=0), 1e-10) )[np.newaxis, :]
            normal2 = normal2 * seg

            depth1 = twoBounce[:, :, 3:6].transpose([2, 0, 1] )
            depth1 = np.ascontiguousarray(depth1 )
            depth1 = depth1 * seg

            twoNormalName = f'{self.data_folder}/imtwoNormalPred{self.camNum}_{ind+1}.npy'
            twoNormals = np.load(twoNormalName )
            normalOpt = twoNormals[:, :, 0:3]
            normalOpt = cv2.resize(normalOpt, (self.imWidth, self.imHeight), interpolation = cv2.INTER_AREA )
            normalOpt = np.ascontiguousarray(normalOpt.transpose([2, 0, 1] ) )
            normalOpt = normalOpt / np.sqrt(np.maximum(np.sum(normalOpt * normalOpt, axis=0), 1e-10) )[np.newaxis, :]
            normalOpt = normalOpt * seg

            env = cv2.cvtColor(cv2.imread(self.opt.env_path, -1), cv2.COLOR_BGRA2BGR)[:, :, ::-1]
            #env = cv2.imread(envFileName, -1)[:, :, ::-1]
            env = cv2.resize(env, (self.opt.envWidth, self.opt.envHeight ), interpolation=cv2.INTER_LINEAR)
            env = np.ascontiguousarray(env )
            env = env / 255
            #env = env.transpose([2, 0, 1]) * imScale * scale
            env = env.transpose([2, 0, 1]).astype(np.float32)

            imNames.append(imName) # TODO: change format to match dataLoader.py
            origins.append(origin)
            lookats.append(lookat)
            ups.append(up)
            normal1s.append(normal1)
            normal2s.append(normal2)
            depth1s.append(depth1)
            depth1VHs.append(depth1[[2]])
            seg1Ints.append(seg[np.newaxis])
            envs.append(env)
            
        pointNameVH = osp.join(self.data_folder, 'visualHullSubd_%d_pts.npy' % self.camNum )
        pointVH = np.load(pointNameVH )
        normalNameVH = osp.join(self.data_folder, 'visualHullSubd_%d_ptsNormals.npy' % self.camNum )
        normalPointVH = np.load(normalNameVH )

        batchDict = {
            'origin': origins,
            'lookat': lookats,
            'up': ups,
            'normal1': normal1s,
            'normal1Opt': normal1s,
            'normal2Opt': normal2s,
            'depth1': depth1s,
            'depth1VH': depth1VHs,
            'seg1Int': seg1Ints,
            'env': envs,
            'point': pointVH, # gt for evaluation only
            'pointVH': pointVH,
            'normalPoint': normalPointVH, # gt for evaluation only
            'normalPointVH': normalPointVH
        }

        for key in batchDict:
            batchDict[key] = torch.tensor(np.array(batchDict[key]), dtype=torch.float32)

        return batchDict

    def loadHDR(self, imName, scale):
        if not osp.isfile(imName):
            print('Error: %s does not exist.' % imName)
            assert(False )
        image = cv2.imread(imName, -1)[:, :, ::-1]
        # image = cv2.resize(image, (self.imWidth, self.imHeight ), interpolation=cv2.INTER_LINEAR)
        image = np.ascontiguousarray(image)
        imMean = np.mean(image)

        if scale is None:
            if self.phase == 'TRAIN':
                scale = (np.random.random() * 0.2 + 0.4) / imMean
            else:
                scale = 0.5 / imMean
        image = np.clip((image*scale), 0, 1).transpose([2, 0, 1])
        return image, scale