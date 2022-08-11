'''
Written by Nathan Neeteson.
A set of utilities for calculating errors and losses on model outputs.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def calc_multiclass_dice_coeff(seg1,seg2):
    # seg1 and seg2 are pytorch tensors with dimensions: (batch,class,width,height)
    # this function assumes that the classes are labelled with integers in a
    # single class channel

    seg1 = seg1.cpu().detach().numpy().flatten()
    seg2 = seg2.cpu().detach().numpy().flatten()

    num_classes = max(max(seg1),max(seg2))

    D = 0

    for c in range(num_classes):

        X = seg1==c
        Y = seg2==c

        D += 2 * np.sum(X&Y) / (np.sum(X) + np.sum(Y) + 1e-8)

    return D/num_classes

def zero_crossings(x):
    # this function takes in a torch tensor and outputs a new torch tensor,
    # which is 1 whereever a zero crossing is detected in the input
    # inputs
    # x: a tensor of shape (batch,channel,height,width)

    # first, calculate the sign of x element-wise
    sgnx = torch.sign(x)

    # the first check is if x is actually 0, if that is the case then that is
    # obviously a zero crossing. we can do this while creating the zero crossing
    # tensor - and we will crop it because the next step will crop a layer
    z = (sgnx==0)[:,:,1:-1,1:-1]

    # now we will check each direct neighbour of each voxel. if the sign of the
    # neighbour does not match, then the voxel is a zero-crossing
    z += sgnx[:,:,1:-1,1:-1] != sgnx[:,:, 1:-1, 2:  ]
    z += sgnx[:,:,1:-1,1:-1] != sgnx[:,:, 1:-1,  :-2]
    z += sgnx[:,:,1:-1,1:-1] != sgnx[:,:, 2:  , 1:-1]
    z += sgnx[:,:,1:-1,1:-1] != sgnx[:,:,  :-2, 1:-1]

    # finally, convert z back to boolean since we just want to detect locations
    # of zero crossings and not count how many occur at a voxel
    return z>0


# this code is from: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
# it still needs to be verified
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.softmax(inputs,dim=1)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/((inputs*inputs).sum() + (targets*targets).sum() + smooth)

        return 1 - dice

def create_approximate_heaviside(epsilon):
    # factory function for creating an approximate heaviside function with a
    # scaling factor of epsilon

    def approximate_heaviside(x):
        return ( 1/2 + (1/np.pi)*torch.atan(x/epsilon) )

    return approximate_heaviside

def create_convert_embeddings_to_predictions(epsilon):
    # factory function to make a function that converts the embeddings for the
    # periosteal and endosteal surfaces into log-probability predictions of
    # whether each voxel is cortical, trabecular, or background

    heaviside = create_approximate_heaviside(epsilon)

    def convert_embeddings_to_predictions(embeddings):

        phi_peri, phi_endo = embeddings[:,0,:,:], embeddings[:,1,:,:]

        # convert surface embeddings into voxel-wise class predictions
        pred_cort = heaviside(-phi_peri)*heaviside(phi_endo)
        pred_trab = heaviside(-phi_endo)
        pred_back = heaviside(phi_peri)

        # stack predictions, normalize and take log to get log-probabilities
        preds = torch.stack((pred_cort,pred_trab,pred_back),dim=1)
        preds = preds / torch.sum(preds,dim=1,keepdim=True)
        preds = torch.log(preds)

        return preds

    return convert_embeddings_to_predictions


def create_calculate_embedding_dice_coefficient(epsilon):
    # factory function to make a function that can calculate the multiclass
    # dice coefficient given the periosteal and endosteal embeddings and the
    # true voxel labels - using the embedding>prediction factory function

    convert_embeddings_to_predictions = \
        create_convert_embeddings_to_predictions(epsilon)

    def calculate_embedding_dice_coefficient(embeddings,labels):

        preds = convert_embeddings_to_predictions(embeddings)

        preds = torch.argmax(preds,dim=1)

        return calc_multiclass_dice_coeff(preds,labels)

    return calculate_embedding_dice_coefficient


def create_approximate_delta(epsilon):
    # factory function for creating an approximate delta function with a
    # threshold of epsilon

    def approximate_delta(x):
        return abs(x) < epsilon

    return approximate_delta

class CurvatureLoss(nn.Module):
    # class defining a loss function based on the curvature in the vicinity of
    # the zero level set of a predicted embedding, for regularization
    def __init__(self, vox_width, H_thresh, device):
        super(CurvatureLoss, self).__init__()

        # small number to prevent divide by zero errors
        self.eps = 1e-8

        # store the curvature threshold as a local attribute
        self.H_thresh = H_thresh

        # store the voxel width as well
        self.vox_width = vox_width

        # we need to construct some convolutional filters to compute the spatial
        # gradients of the embedding fields. we will use 4th order central diff

        # NOTE: the way that the images get imported from the vtk reader
        # orients the dimensions so that the image samples are: (b,c,h,w),
        # so at some point it would be worth checking if these kernels need to
        # be transposed so that 'dx' corresponds to width and 'dy' to height.
        # it doesn't actually matter right now since the loss measures are
        # invariant to 90 degree axial rotations, but it would be better to
        # be consistent and labelling these correctly in case in the future
        # they get copied and used for something else.

        '''
        ddx_kernel = np.array(
            [
                [   0,      0,       0,      0,      0       ],
                [   0,      0,       0,      0,      0       ],
                [   1/12,   -2/3,    0,      2/3,    -1/12   ],
                [   0,      0,       0,      0,      0       ],
                [   0,      0,       0,      0,      0       ]
            ]
        )
        '''

        ddx_kernel = np.array(
            [
                [   0,  0,   0],
                [-1/2,  0, 1/2],
                [   0,  0,   0]
            ]
        )

        '''
        ddy_kernel = np.array(
            [
                [   0,      0,      1/12,   0,      0   ],
                [   0,      0,      -2/3,   0,      0   ],
                [   0,      0,      0,      0,      0   ],
                [   0,      0,      2/3,    0,      0   ],
                [   0,      0,      -1/12,  0,      0   ]
            ]
        )
        '''

        ddy_kernel = np.array(
            [
                [0, -1/2, 0],
                [0, 0,    0],
                [0,  1/2, 0]
            ]
        )

        '''
        d2dx2_kernel = np.array(
            [
                [   0,      0,      0,      0,      0       ],
                [   0,      0,      0,      0,      0       ],
                [   -1/12,  4/3,    -5/2,   4/3,    -1/12   ],
                [   0,      0,      0,      0,      0       ],
                [   0,      0,      0,      0,      0       ]
            ]
        )
        '''

        d2dx2_kernel = np.array(
            [
                [0,  0, 0],
                [1, -2, 1],
                [0,  0, 0]
            ]
        )

        '''
        d2dy2_kernel = np.array(
            [
                [   0,      0,      -1/12,  0,      0   ],
                [   0,      0,      4/3,    0,      0   ],
                [   0,      0,      5/2,    0,      0   ],
                [   0,      0,      4/3,    0,      0   ],
                [   0,      0,      -1/12,  0,      0   ]
            ]
        )
        '''

        d2dy2_kernel = np.array(
            [
                [0,  1, 0],
                [0, -2, 0],
                [0,  1, 0]
            ]
        )

        # credit: http://www.holoborodko.com/pavel/2014/11/04/computing-mixed-derivatives-by-finite-differences/
        '''
        d2dxdy_kernel = np.array(
            [
                [   -1/144, -8/144, 0,  8/144,  -1/144  ],
                [   -8/144, 64/144, 0,  64/144, 8/144   ],
                [   0,      0,      0,  0,      0       ],
                [   8/144,  64/144, 0,  64/144, -8/144  ],
                [   -1/144, 8/144,  0,  -8/144, -1/144  ]
            ]
        )
        '''

        d2dxdy_kernel = np.array(
            [
                [ 1/4,  0, -1/4],
                [   0,  0,    0],
                [-1/4,  0,  1/4]
            ]
        )

        # now we can construct torch tensors of the appropriate dimension to
        # serve as the kernels for conv2d operations that will apply these
        # spatial gradients to the embedding fields

        self.ddx = torch.tensor(ddx_kernel).unsqueeze(0).unsqueeze(0).float().to(device)
        self.ddy = torch.tensor(ddy_kernel).unsqueeze(0).unsqueeze(0).float().to(device)

        self.d2dx2 = torch.tensor(d2dx2_kernel).unsqueeze(0).unsqueeze(0).float().to(device)
        self.d2dy2 = torch.tensor(d2dy2_kernel).unsqueeze(0).unsqueeze(0).float().to(device)

        self.d2dxdy = torch.tensor(d2dxdy_kernel).unsqueeze(0).unsqueeze(0).float().to(device)

    def forward(self, phi):

        # compute all the required spatial gradients
        dphidx = F.conv2d(phi,self.ddx)/self.vox_width
        dphidy = F.conv2d(phi,self.ddy)/self.vox_width
        d2phidx2 = F.conv2d(phi,self.d2dx2)/(self.vox_width**2)
        d2phidy2 = F.conv2d(phi,self.d2dy2)/(self.vox_width**2)
        d2phidxdy = F.conv2d(phi,self.d2dxdy)/(self.vox_width**2)

        # compute the numerator and denominator of the curvature expression
        H_numerator = d2phidx2*torch.pow(dphidy,2) - 2*dphidy*dphidx*d2phidxdy + d2phidy2*torch.pow(dphidx,2)
        H_denominator = torch.pow((torch.pow(dphidx,2) + torch.pow(dphidy,2)),3/2)

        # compute the curvature field
        H = H_numerator / (H_denominator + self.eps)

        # then take the square of the curvature field divided by the curvature
        # threshold, subtract 1 from that, and apply a rectified linear unit.
        # we do this because we do not want there to be any penalty for curvature
        # that is within acceptable bounds, only excess curvature
        H_relu = F.relu(torch.pow(H/(self.H_thresh+self.eps),2)-1)

        # detect the zero crossings in phi
        phi_zero = zero_crossings(phi)

        # the final operation is to compute the overall loss by calculating the
        # average curvature on the contour defined by the zero level set,
        # where the zero level set has been determined as the voxels where
        # the level set crosses zero

        return torch.sum(phi_zero*H_relu) / (torch.sum(phi_zero) + self.eps)



class MagnitudeGradientLoss(nn.Module):
    # class defining a loss function based on the integrated magnitude of the
    # gradient of a predicted embedding, for regularization
    def __init__(self, vox_width, device):
        super(MagnitudeGradientLoss, self).__init__()

        # small number to prevent divide by zero errors
        self.eps = 1e-8

        # store the voxel width as well
        self.vox_width = vox_width

        # we need to construct some convolutional filters to compute the spatial
        # gradients of the embedding fields. we will use 4th order central diff

        '''
        ddx_kernel = np.array(
            [
                [   0,      0,       0,      0,      0       ],
                [   0,      0,       0,      0,      0       ],
                [   1/12,   -2/3,    0,      2/3,    -1/12   ],
                [   0,      0,       0,      0,      0       ],
                [   0,      0,       0,      0,      0       ]
            ]
        )
        '''

        ddx_kernel = np.array(
            [
                [   0,  0,   0],
                [-1/2,  0, 1/2],
                [   0,  0,   0]
            ]
        )

        '''
        ddy_kernel = np.array(
            [
                [   0,      0,      1/12,   0,      0   ],
                [   0,      0,      -2/3,   0,      0   ],
                [   0,      0,      0,      0,      0   ],
                [   0,      0,      2/3,    0,      0   ],
                [   0,      0,      -1/12,  0,      0   ]
            ]
        )
        '''

        ddy_kernel = np.array(
            [
                [0, -1/2, 0],
                [0, 0,    0],
                [0,  1/2, 0]
            ]
        )

        # now we can construct torch tensors of the appropriate dimension to
        # serve as the kernels for conv2d operations that will apply these
        # spatial gradients to the embedding fields

        self.ddx = torch.tensor(ddx_kernel).unsqueeze(0).unsqueeze(0).float().to(device)
        self.ddy = torch.tensor(ddy_kernel).unsqueeze(0).unsqueeze(0).float().to(device)

    def forward(self, phi):

        # compute the required spatial gradients
        dphidx = F.conv2d(phi,self.ddx)/self.vox_width
        dphidy = F.conv2d(phi,self.ddy)/self.vox_width

        # calculate the magnitude gradient field of phi
        maggradphi = torch.sqrt(torch.pow(dphidx,2)+torch.pow(dphidy,2)+self.eps)

        # return the average magnitude gradient in the field as the loss
        return torch.sum(maggradphi) / (torch.sum(torch.ones_like(maggradphi))+self.eps)

class MagnitudeGradientSDTLoss(nn.Module):
    # class defining a loss function based on the integrated magnitude of the
    # gradient of a predicted embedding, for regularization
    # this function assumes that you want to regularize your embedding towards
    # being a proper signed distance transform and as such the construction
    # of the loss term forces the magnitude of the gradient of phi to one
    def __init__(self, vox_width, device):
        super(MagnitudeGradientSDTLoss, self).__init__()

        # small number to prevent divide by zero errors
        self.eps = 1e-8

        # store the voxel width as well
        self.vox_width = vox_width

        # we need to construct some convolutional filters to compute the spatial
        # gradients of the embedding fields. we will use 4th order central diff

        '''
        ddx_kernel = np.array(
            [
                [   0,      0,       0,      0,      0       ],
                [   0,      0,       0,      0,      0       ],
                [   1/12,   -2/3,    0,      2/3,    -1/12   ],
                [   0,      0,       0,      0,      0       ],
                [   0,      0,       0,      0,      0       ]
            ]
        )
        '''

        ddx_kernel = np.array(
            [
                [   0,  0,   0],
                [-1/2,  0, 1/2],
                [   0,  0,   0]
            ]
        )

        '''
        ddy_kernel = np.array(
            [
                [   0,      0,      1/12,   0,      0   ],
                [   0,      0,      -2/3,   0,      0   ],
                [   0,      0,      0,      0,      0   ],
                [   0,      0,      2/3,    0,      0   ],
                [   0,      0,      -1/12,  0,      0   ]
            ]
        )
        '''

        ddy_kernel = np.array(
            [
                [0, -1/2, 0],
                [0, 0,    0],
                [0,  1/2, 0]
            ]
        )

        # now we can construct torch tensors of the appropriate dimension to
        # serve as the kernels for conv2d operations that will apply these
        # spatial gradients to the embedding fields

        self.ddx = torch.tensor(ddx_kernel).unsqueeze(0).unsqueeze(0).float().to(device)
        self.ddy = torch.tensor(ddy_kernel).unsqueeze(0).unsqueeze(0).float().to(device)

    def forward(self, phi):

        # compute the required spatial gradients
        dphidx = F.conv2d(phi,self.ddx)/self.vox_width
        dphidy = F.conv2d(phi,self.ddy)/self.vox_width

        # calculate the magnitude gradient field of phi
        maggradphi = torch.sqrt(torch.pow(dphidx,2)+torch.pow(dphidy,2)+self.eps)

        # then take the square of the log of the magnitude of the gradient of phi
        sq_log_maggradphi = torch.pow(torch.log(maggradphi),2)

        # then get a map of where phi is not a zero crossing
        phi_not_zero = torch.logical_not(zero_crossings(phi))

        # then the loss will be the average of the square of the log of the
        # magnitude of the gradient of phi, everywhere that is not a zero crossing
        return torch.sum(phi_not_zero*sq_log_maggradphi) / (torch.sum(phi_not_zero)+self.eps)

class HRpQCTEmbeddingNLLLoss(nn.Module):

    def __init__(self, heaviside_epsilon):
        super(HRpQCTEmbeddingNLLLoss, self).__init__()

        self.convert_embeddings_to_predictions = \
            create_convert_embeddings_to_predictions(heaviside_epsilon)
        self.loss = nn.NLLLoss()

    def forward(self,embeddings,labels):

        preds = self.convert_embeddings_to_predictions(embeddings)

        return self.loss(preds,labels.squeeze(1))

class HRpQCTEmbeddingCombinedRegularizationLoss(nn.Module):

    def __init__(self,loss):
        super(HRpQCTEmbeddingCombinedRegularizationLoss, self).__init__()

        self.loss = loss

    def forward(self,embeddings,labels):

        phi_peri, phi_endo = embeddings[:,[0],:,:], embeddings[:,[1],:,:]

        return self.loss(phi_peri) + self.loss(phi_endo)
