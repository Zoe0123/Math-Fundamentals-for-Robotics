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
# after removk
inds = []

with open('regions_red_14.txt') as f:
    for line in f:
        inds.append(np.array(line.split()).astype(int))

# from region_inds_raw import a1,a2,a3,a4,a5,a6,a7, a8

# inds = [a1,a2,a3,a4,a5,a6,a7,a8]



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
test_offset_gt = torch.tensor(test_gt - all_mesh[0:1], device='cuda')

test_reconsts = np.zeros((len(paths[1000:]), 5023,3 ))

print(len(inds))

# wts = [1.,1,1,1,1,1,1,1]

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
    pca = PCA(n_components=30)
    pca.fit(train_region)



    components = pca.components_.reshape(-1, n, 3)

    test_region = pca.transform(test_region)
    print(components.shape)
    print(test_region.shape)

    pcas.append(pca)
    test_regions_params.append(test_region)
    templ_vs.append(templ_v)
    
    # obj += np.linalg.norm(new-region[:1000])

bases = [torch.tensor(pca.components_).cuda() for pca in pcas]
params = [torch.nn.Parameter(torch.tensor(p).cuda()) for p in test_regions_params]

opt = torch.optim.AdamW(params, lr = 1e-2)
n_iter = 400
beta = 0.5

losses = []
for it in range(n_iter):
    opt.zero_grad()
    loss = 0
    loss_rec = 0

    for r,ind in enumerate(inds):
        templ_v = templ_vs[r]
        test_param = params[r] # N sample, n comp, 3
        components = bases[r]  # N comp, n ind region, 3
        n = len(ind)

        # print('loop',components.shape, test_param.shape)
        test_res = torch.sum(components[:].reshape(-1, n, 3)*test_param[:,:,None, None], dim=1)
        # print('gt offset', test_offset_gt[:,ind,:].shape, test_res.shape)
        loss_rec += torch.sum(torch.norm(test_offset_gt[:,ind,:] - test_res, dim=-1))

    loss_blend = 0

    for ri,ind_i in enumerate(inds):
        for rj,ind_j in enumerate(inds):

            if ri != rj:

                

                inds_common, ind_ic, ind_jc = np.intersect1d(ind_i, ind_j, assume_unique=False, return_indices=True)
                # print(ind_i, ind_j, inds_common)
                # print('INDSSS',ind_ic, ind_jc)
                n = len(ind_ic)
                if len(ind_ic) > 0:
                    test_param_i = params[ri]
                    components_i = bases[ri].reshape(-1, len(ind_i), 3).permute((1,0, 2))[torch.tensor(ind_ic).cuda()].permute((1,0,2))

                    # print(params[ri].data.shape, params[rj].data.shape)

                    test_param_j = params[rj]
                    components_j = bases[rj].reshape(-1, len(ind_j), 3).permute((1,0, 2))[torch.tensor(ind_jc).cuda()].permute((1,0,2))


                    # print('HERE',bases[rj].reshape(-1, len(ind_j), 3).permute((1,0, 2)).shape, ind_jc.shape)
                    # print(ind_jc, ind_ic)
                    # print(bases[rj].reshape(-1, len(ind_j), 3).permute((1,0, 2)))
                    # components_j = torch.take_along_dim(bases[rj].reshape(-1, len(ind_j), 3).permute((1,0, 2)), indices=torch.tensor(ind_jc).cuda(), dim=0).permute((1,0,2))

                    

                    # print('loop2',components_j.shape, test_param_j.shape)
                    # print('loop2_i',components_i.shape, test_param_i.shape)


                    loss_blend += torch.sum(torch.norm(torch.sum(components_j*test_param_j[:,:,None, None], dim=1) - torch.sum(components_i[:]*test_param_i[:,:,None, None], dim=1)))
            
    loss = loss_rec + loss_blend * beta

    print('it', it, 'loss_rec', loss_rec,'loss_blend', loss_blend)


    # pytorch minimizes losses, we want to maximize ours -> * (-1)
    loss.backward()
    opt.step()
    losses.append(loss.item())

test_reconsts = np.zeros((len(paths[1000:]), 5023,3 ))


for r,ind in enumerate(inds):
    templ_v = templ_vs[r]
    test_param = params[r] # N sample, n comp, 3
    components = bases[r]  # N comp, n ind region, 3

    n = len(ind)

    test_res = np.sum(components[:].reshape(-1, n, 3).detach().cpu().numpy()*test_param[:,:,None, None].detach().cpu().cpu().numpy(), axis=1)+templ_v.reshape((n,3))

    test_reconsts[:, ind,:] = test_res


for i in range(len(test_gt)):

    if i==554:
        for j in range(1):
            mesh = Trimesh(
                        vertices=test_reconsts[i],
                        faces=obj['faces'])
            mesh.export('./regiontest/test554_smoothchang.ply')

            mesh = Trimesh(
                        vertices=test_gt[i],
                        faces=obj['faces'])
            mesh.export('./regiontest/test554_gt.ply'.format(i))
