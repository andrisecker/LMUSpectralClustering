#!/usr/bin/python
# -*- coding: utf8 -*-
# created by Ajayrama Kumaraswamy
# modified by András Ecker 07/2015

import numpy as np
from scipy import linalg
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigs


#Linear Ordering Based On Spectral Clustering
#Ding, C., & He, X. (2004). Linearized Cluster Assignment via Spectral Ordering.
# In Proceedings of the 21st International Conference on Machine Learning.


# @profile
def linearOrdering(matW):
    '''
    Uses spectral clustering algorithm to return a linear ordering
    :param matW: similiarityMatrix: square numpy array for similarities
    :return: numpy array containing same number of elements as rows as the input.
    '''

    sh = np.shape(matW)

    assert sh[0] == sh[1]
    assert (matW == matW.T).all()

    invRootMatD = np.diagflat(1/(np.sqrt(matW.sum(axis=1))))  # invRootMatD = D^-1/2

    matA = np.dot(invRootMatD, np.dot(matW, invRootMatD))  # matA = D^-1/2*W*D^-1/2

    w, v = eigs(matA, k=2, which='LM')

    z1 = v[:, 1]  # second largest eigenvector

    q1 = np.dot(invRootMatD, z1)

    ordering = np.argsort(q1)  # indices that sort q1

    matW_rowOrdered = matW[ordering, :]

    matW_ordered = matW_rowOrdered[:, ordering]

    return ordering, matW_ordered


# @profile
def calcClusterCrossing(matW, m):
    '''
    Calculates the cluster crossing
    :param matW: square numpy array representing the similarity matrix
    :param m: bandwidth along the antidiagonals to use.
    :return: a list of (matW.shape - 1) elements containing cluster crossings.
    '''

    sh = np.shape(matW)

    assert sh[0] == sh[1]
    assert (matW == matW.T).all()

    n = sh[0]

    matWRev = matW[:, ::-1]

    crossings = []

    for offset in np.arange(n - 2, -(n - 1), -2):

        dia = np.diagonal(matWRev, offset)
        #len(dia) is even
        mid = len(dia) / 2
        sigi = sum(dia[mid: min(mid + m, len(dia) - 1)])
        sigim = 0
        sigip = 0
        if offset > -(n - 2):
            diam = np.diagonal(matWRev, offset - 1)
            # len(diam) is odd; need to exclude the element on the diagonal of matW (antidiagonal of matWRev)
            mid = len(diam) / 2 + 1
            sigim = sum(diam[mid: min(mid + m, len(dia) - 1)])

        if offset < n - 2:
            diap = np.diagonal(matWRev, offset + 1)
            # len(diap) is odd; need to exclude the element on the diagonal of matW (antidiagonal of matWRev)
            mid = len(diap) / 2 + 1
            sigip = sum(diap[mid: min(mid + m, len(dia) - 1)])

        crossing = 0.5 * sigi + 0.25 * sigim + 0.25 * sigip

        if len(dia) <= 2 * m:

            crossing *= 2 * m / float(len(dia))

        crossings.append(crossing)

    return crossings + [0]

# @profile
def optSig(matDist):
    '''
    Based on distance distribution finds the optimal sigma of the Gaussian kernel
    \sigma_i = e^{\frac{||maxd_i^2-mind_i^2||}{2ln\frac{maxd_i^2}{mind_i^2}}}
    \sigma = average of \sigma_i-s
    :param matDist: square numpy array representing the distance matrix
    :return: optimal sigma
    '''

    matDist[matDist == 0] = np.nan

    maxV = np.nanmax(matDist, axis=1)
    minV = np.nanmin(matDist, axis=1)

    diff = maxV**2 - minV**2
    frac = maxV**2/minV**2

    sigV = np.sqrt(diff/(2*np.log(frac)))

    return np.mean(sigV)



# @profile
def calcSimMatrix(mat):
    '''
    Creates a SimilarityMatrix from data points, based on euclidean distance
    and a gaussian kernel with standard deviation
    :param mat: numpy array containing the data points
    :return:numpy array: similarity matrix
    '''

    matDist = squareform(pdist(mat, lambda u, v: linalg.norm(u-v)))

    sig = optSig(matDist)

    print 'optimal sig: ', sig

    matDist = np.nan_to_num(matDist)

    matW = np.exp(-(np.power(matDist, 2))/(2*sig**2))

    return matW


# @profile
def spectralClustering(mat, m):
    '''
    Calculate SimilarityMartix and apply SpectralClustering
    :param mat: numpy array containing the data points
    :param m: bandwidth along the antidiagonals to use.
    :return: ordered similarityMatrix (matW_ordered),
             a list of (matW.shape) elements containing cluster crossings (crossing)
             the ordering (ordering)
    '''

    matW = calcSimMatrix(mat)

    ordering, matW_ordered = linearOrdering(matW)

    crossing = calcClusterCrossing(matW_ordered, m)

    return matW_ordered, crossing, ordering

