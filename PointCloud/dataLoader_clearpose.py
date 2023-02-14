from torch.utils.data import Dataset
from scipy.io import loadmat
import cv2
import os.path as osp
import numpy as np

class BatchLoader(Dataset):
    def __init__(self, opt):

        self.dataRoot = opt.dataRoot
        self.imHeight = opt.imHeight
        self.imWidth = opt.imWidth
        self.phase = 'TEST'

        self.img_idx = list(range(opt.img_idx_start, opt.img_idx_end+1, opt.img_idx_step))

        self.camera_pose = self.load_camera(opt.meta_file)

    def load_camera(self, meta_file):
        metadata = loadmat(meta_file)
        self.camDict = {}
        for i in self.img_idx:
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
            self.camDict[i] = paramDict

    def __len__(self):
        return len(self.img_idx)

    def __getitem__(self, ind):
        # normalize the normal vector so that it will be unit length
        imNames = []
        seg1s = []
        seg2VHs = []
        normal1VHs = []
        normal2VHs = []
        normal1s = []
        depth1s = []
        ims = []
        imEs = []

        origins = []
        lookats = []
        ups = []

        envs = []

        for img_idx in self.img_idx:
            imName = self.camDict[img_idx]['imgName']
            imE, imScale = self.loadHDR(imName, imScale)
            im = cv2.imread(f'{self.dataRoot}/{img_idx:06d}-color.png')
            seg = cv2.imread(f'{self.dataRoot}/{img_idx:06d}-label-predict.png', -1)
            seg[seg != 0] = 1
            im = imE * seg
            origin, lookat, up = self.camDict[img_idx]['Origin'], self.camDict[img_idx]['Target'], self.camDict[img_idx]['Up'] # check target == lookat
            env = cv2.imread('TransparentShapeReconstruction/RealData/Envmaps/real/env_4140.png') # TODO: change relative path if needed
            # TODO: normal1, normal1VH, normal2VH, seg1, seg2VH should be from renderer using visual hull mesh
            normal1 = ((cv2.imread(f'{self.dataRoot}/{img_idx:06d}-normal_true.png').astype(np.float32) / 255.) - 0.5) * 2
            normal1VH = normal1
            normal2VH = ((cv2.imread(f'{self.dataRoot}/{img_idx:06d}-normal_true.png').astype(np.float32) / 255.) - 0.5) * 2 # TODO: change suffix of line 69 and 67 to visual hull rendered normals
            seg1 = seg
            seg2VH = seg
            depth1 = cv2.imread(f'{self.dataRoot}/{img_idx:06d}-depth_pred.png', -1)

            imNames.append(imName) # TODO: change format to match dataLoader.py
            ims.append(im)
            imEs.append(imE)
            origins.append(origin)
            lookats.append(lookat)
            ups.append(up)
            normal1s.append(normal1)
            normal1VHs.append(normal1VH)
            normal2VHs.append(normal2VH)
            depth1s.append(depth1)
            seg1s.append(seg1)
            seg2VHs.append(seg2VH)
            envs.append(env)

        batchDict = {
            'name': imNames,
            'im':  ims,
            'imE': imEs,
            'origin': origins,
            'lookat': lookats,
            'up': ups,
            'normal1': normal1s,
            'normal1VH': normal1VHs,
            'normal2VH': normal2VHs,
            'depth1': depth1s,
            'seg2VH': seg2VHs,
            'seg1': seg1s,
            'env': envs,
        }

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