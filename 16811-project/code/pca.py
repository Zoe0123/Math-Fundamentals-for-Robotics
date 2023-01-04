

import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
import trimesh
from trimesh import Trimesh


import os
import numpy as np

from utils import *

import torch

data_dir = './Helen'

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

Vs = np.stack(Vs)

Vs_flat = Vs.reshape(-1, 5023*3)

templ_v = Vs_flat[0]
offset_vs = Vs_flat - templ_v

train_vs = offset_vs[:1000]
test_vs = offset_vs[1000:]
test_vs2 = offset_vs[1000:]

# offset_vs = torch.tensor(offset_vs).cuda()
# U,S,V = torch.pca_lowrank(offset_vs)

from sklearn.decomposition import PCA
pca = PCA(n_components=50)
pca.fit(train_vs)

components = pca.components_.reshape(-1, 5023, 3)

vs_new = pca.transform(test_vs)
print(vs_new.shape)
print(components.shape)

# print(vs_new[:10])

# for c in range(50):

#     mesh = Trimesh(
#             vertices=components[c]+templ_v.reshape((5023,3)),
#             faces=obj['faces'])
#     mesh.export('./components/comp{}.ply'.format(c))

# print('PCA out', components.shape)

for i in range(len(test_vs)):
    mesh = Trimesh(
                vertices=np.sum(components[:]*vs_new[i][:,None, None], axis=0)+templ_v.reshape((5023,3)),
                faces=obj['faces'])
    mesh.export('./test_unseen/test{}.ply'.format(i))

    mesh = Trimesh(
                vertices=test_vs2[i].reshape((5023,3))+templ_v.reshape((5023,3)),
                faces=obj['faces'])
    mesh.export('./test_unseen/test{}_gt.ply'.format(i))
