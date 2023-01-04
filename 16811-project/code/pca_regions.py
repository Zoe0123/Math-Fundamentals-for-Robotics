import trimesh
# from pytorch3d.structures import Meshes
# from pytorch3d.io import load_obj
import trimesh
from trimesh import Trimesh


import os
import numpy as np

from utils import *

import torch

data_dir = './Helen'

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
obj = 0

for ind in inds:
    n = len(ind)
    region = np.stack(Vs)[:, ind, :] 

    region_flat = region.reshape(-1, n*3)

    templ_v = region_flat[0]
    offset_region = region_flat - templ_v

    train_region = offset_region[:1000]

    # offset_region = torch.tensor(offset_region).cuda()
    # U,S,V = torch.pca_lowrank(offset_region)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=50)
    pca.fit(train_region)

    components = pca.components_.reshape(-1, n, 3)

    region_new = pca.transform(train_region)
    print(components.shape)
    print(region_new.shape)
    
    # not sure if this is the correct objective?
    new = np.sum(components[:]*region_new[:,:,None, None], axis=1)+templ_v.reshape((n,3))
    obj += np.linalg.norm(new-region[:1000])

obj /= len(inds)
print(obj)


# TODO: test

Vs = np.stack(Vs) # 1575, 5023, 3

Vs_flat = Vs.reshape(-1, 5023*3)

templ_v = Vs_flat[0]
offset_vs = Vs_flat - templ_v

test_vs = offset_vs[1000:]
test_vs2 = offset_vs[1000:]



# for i in range(len(test_vs)):
#     mesh = Trimesh(
#                 vertices=np.sum(components[:]*vs_new[i][:,None, None], axis=0)+templ_v.reshape((5023,3)),
#                 faces=obj['faces'])
#     mesh.export('./test_unseen/test{}.ply'.format(i))

#     mesh = Trimesh(
#                 vertices=test_vs2[i].reshape((5023,3))+templ_v.reshape((5023,3)),
#                 faces=obj['faces'])
#     mesh.export('./test_unseen/test{}_gt.ply'.format(i))
