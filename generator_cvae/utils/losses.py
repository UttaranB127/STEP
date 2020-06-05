import numpy as np
import torch
import torch.nn as nn
from utils.common import *


def between_frame_loss(gait1, gait2, thres=0.01):
    g1 = gait1.permute(0, 2, 3, 1, 4).contiguous().view(gait1.shape[0], gait1.shape[2], gait1.shape[1]*gait1.shape[3])
    g2 = gait2.permute(0, 2, 3, 1, 4).contiguous().view(gait2.shape[0], gait2.shape[2], gait2.shape[1]*gait2.shape[3])
    num_batches = g1.shape[0]
    num_tsteps = g2.shape[1]
    mid_tstep = np.int(num_tsteps / 2) - 1
    loss = nn.functional.mse_loss(g1, g2)
    for bidx in range(num_batches):
        for tidx in range(num_tsteps):
            loss += nn.functional.mse_loss(g1[bidx, tidx, :]-g1[bidx, 0, :],
                                           g2[bidx, tidx, :]-g2[bidx, 0, :])
            loss += nn.functional.mse_loss(g1[bidx, tidx, :]-g1[bidx, mid_tstep, :],
                                           g2[bidx, tidx, :]-g2[bidx, mid_tstep, :])
            loss += nn.functional.mse_loss(g1[bidx, tidx, :]-g1[bidx, -1, :],
                                           g2[bidx, tidx, :]-g2[bidx, -1, :])
            for vidx in range(g1.shape[2]):
                if tidx > 0:
                    loss += nn.functional.mse_loss(g1[bidx, tidx, vidx] - g1[bidx, tidx-1, vidx],
                                                   g2[bidx, tidx, vidx] - g2[bidx, tidx-1, vidx])
                if tidx > 1:
                        loss += nn.functional.mse_loss(g1[bidx, tidx, vidx] -
                                                       2*g1[bidx, tidx-1, vidx] + g1[bidx, tidx-2, vidx],
                                                       g2[bidx, tidx, vidx] -
                                                       2 * g2[bidx, tidx - 1, vidx] + g2[bidx, tidx - 2, vidx])
                # loss += nn.functional.l1_loss(g2[bidx, tidx, 5], g2[bidx, tidx-1, 5]+thres/2)
                # loss += nn.functional.l1_loss(g2[bidx, tidx, 6], g2[bidx, tidx-1, 6]+thres)
                # loss += nn.functional.l1_loss(g2[bidx, tidx, 7], g2[bidx, tidx-1, 7]+thres/3)
                # loss += nn.functional.l1_loss(g2[bidx, tidx, 8], g2[bidx, tidx-1, 8]+thres/2)
                # loss += nn.functional.l1_loss(g2[bidx, tidx, 9], g2[bidx, tidx-1, 9]+thres)
                # loss += nn.functional.l1_loss(g2[bidx, tidx, 10], g2[bidx, tidx-1, 10]+thres/3)
                # loss += nn.functional.l1_loss(g2[bidx, tidx, 11], g2[bidx, tidx-1, 11]+thres/2)
                # loss += nn.functional.l1_loss(g2[bidx, tidx, 12], g2[bidx, tidx-1, 12]+thres)
                # loss += nn.functional.l1_loss(g2[bidx, tidx, 13], g2[bidx, tidx-1, 13]+thres/3)
                # loss += nn.functional.l1_loss(g2[bidx, tidx, 14], g2[bidx, tidx-1, 14]+thres/2)
                # loss += nn.functional.l1_loss(g2[bidx, tidx, 15], g2[bidx, tidx-1, 15]+thres)
    return loss


def affective_loss(gait1, gait2, anchor_weight=1., aff_weight=1.):
    g1 = gait1.permute(0, 2, 3, 1, 4).contiguous().view(gait1.shape[0], gait1.shape[2], gait1.shape[3], gait1.shape[1])
    g2 = gait2.permute(0, 2, 3, 1, 4).contiguous().view(gait2.shape[0], gait2.shape[2], gait2.shape[3], gait2.shape[1])
    num_batches = g1.shape[0]
    num_tsteps = g2.shape[1]
    mid_tstep = np.int(num_tsteps/2)-1
    loss = nn.functional.l1_loss(g1, g2)
    for bidx in range(num_batches):
        for tidx in range(num_tsteps):
            loss += anchor_weight * nn.functional.l1_loss(g1[bidx, tidx, :, :]-g1[bidx, 0, :, :],
                                                          g2[bidx, tidx, :, :]-g2[bidx, 0, :, :])
            loss += anchor_weight * nn.functional.l1_loss(g1[bidx, tidx, :, :]-g1[bidx, mid_tstep, :, :],
                                                          g2[bidx, tidx, :, :]-g2[bidx, mid_tstep, :, :])
            loss += anchor_weight * nn.functional.l1_loss(g1[bidx, tidx, :, :]-g1[bidx, -1, :, :],
                                                          g2[bidx, tidx, :, :]-g2[bidx, -1, :, :])
    # also take difference between each t-th and t+n-th frame
    # af1 = get_affective_features(g1.detach().cpu().numpy())[:, :, 48:79]
    # af2 = get_affective_features(g2.detach().cpu().numpy())[:, :, 48:79]
    # loss += aff_weight * nn.functional.mse_loss(torch.from_numpy(af1).float().to(gait1.get_device()),
    #                                             torch.from_numpy(af2).float().to(gait2.get_device()))
    return to_var(torch.FloatTensor([loss]))
