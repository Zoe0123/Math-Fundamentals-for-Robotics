import trimesh
# from pytorch3d.structures import Meshes
# from pytorch3d.io import load_obj
import trimesh
from trimesh import Trimesh


import os
import numpy as np

from utils import *

import torch

data_dir = '../Helen'

# load inds of each 8 region (list of arrays)
inds = []

with open('regions.txt') as f:
    for line in f:
        inds.append(np.array(line.split()).astype(int))

Vs = []

paths = []

for root, subdirectories, files in os.walk(data_dir):

    for f in files:
        
        path = os.path.join(root, f)
        paths.append(path)

for p in paths:

    f = open(p,'rb')
    obj = trimesh.exchange.ply.load_ply(f, maintain_order=True)
    verts, faces = obj['vertices'], obj['faces']
    Vs.append(verts)

# TODO: 1. use pytorch and gradient update; 2. include boundary constraint
# train 
# obj = 0
all_mesh = np.stack(Vs)
test_gt = all_mesh[1000:]

test_reconsts = np.zeros((len(paths[1000:]), 5023,3 ))

print(len(inds))

wts = [1.,1,1,1,1,1,1,1]

pcas = []
templ_vs = []

test_regions_params  = []

for ind in inds:
    n = len(ind)
    region = all_mesh[:, ind, :] 

    region_flat = region.reshape(-1, n*3)

    templ_v = region_flat[0]
    offset_region = region_flat - templ_v

    train_region = offset_region[:1000]
    test_region = offset_region[1000:]

    # offset_region = torch.tensor(offset_region).cuda()
    # U,S,V = torch.pca_lowrank(offset_region)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=100)
    pca.fit(train_region)



    components = pca.components_.reshape(-1, n, 3)

    test_region = pca.transform(test_region)
    print(components.shape)
    print(test_region.shape)

    pcas.append(pca)
    test_regions_params.append(test_region)
    templ_vs.append(templ_v)
    
    # not sure if this is the correct objective?
    test_res = np.sum(components[:]*test_region[:,:,None, None]*1.3, axis=1)+templ_v.reshape((n,3))

    test_reconsts[:, ind,:] = test_res
    
    
    # obj += np.linalg.norm(new-region[:1000])

test_reconsts_changeeach = []
factor = 0.7

for j in range(9):
    test_reconsts = np.zeros((len(paths[1000:]), 5023,3 ))
    for i, ind in enumerate(inds):

        n = len(ind)

        test_param = test_regions_params[i]

        templ_v = templ_vs[i]
        components = pcas[i].components_.reshape(-1, len(ind), 3)

        if i == j:
            test_param[:,0] = test_param[:,0]*factor
            test_res = np.sum(components[:]*test_param[:,:,None, None], axis=1)+templ_v.reshape((n,3))
        else:
            test_res = np.sum(components[:]*test_param[:,:,None, None], axis=1)+templ_v.reshape((n,3))

        test_reconsts[:, ind,:] = test_res

    test_reconsts_changeeach.append(test_reconsts)


# obj /= len(inds)
# print(obj)



for i in range(len(test_gt)):

    if i==554:
        for j in range(9):
            mesh = Trimesh(
                        vertices=test_reconsts_changeeach[j][i],
                        faces=obj['faces'])
            mesh.export('./regiontest/test554_07chang{}.ply'.format(j))

            mesh = Trimesh(
                        vertices=test_gt[i],
                        faces=obj['faces'])
            mesh.export('./regiontest/test554_gt.ply'.format(i))
