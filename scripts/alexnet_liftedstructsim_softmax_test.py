#!/usr/bin/python
# coding: utf-8

import _init_paths
import numpy as np
import matplotlib.pyplot as plt
import caffe
import scipy.io as io
import sys
import lmdb
import math

# compute recall at K
def recall_at_K(labels, S, K):
    num = S.shape[0]
    num_K = len(K)
    num_correct = np.zeros(num_K)
    for i in xrange(num):
        index = np.argsort(S[i,:])[::-1]
        for j in xrange(num_K):
            knn_index = index[:K[j]]
            ids = np.where(labels[knn_index] == labels[i])[0]
            if len(ids) > 0:
                num_correct[j] += 1

    for i in xrange(num_K):
        recall = float(num_correct[i]) / float(num)
        print 'K: %d, Recall: %.3f\n' % (K[i], recall)


MODEL_FILE = 'models/AlexNet/test_liftedstructsim_softmax_embed64.prototxt'
CAFFE_MODEL = 'output/alexnet_liftedstructsim_softmax_embed64_iter_65.caffemodel'
MEAN_FILE = 'data/imagenet_mean.binaryproto'
LMDB_FILENAME = 'data/04004475_lmdb'
embedding_dimension = 64
batchsize = 128

caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net(MODEL_FILE, CAFFE_MODEL, caffe.TEST)

# open lmdb
lmdb_env = lmdb.open(LMDB_FILENAME)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()

# count number of images
num_imgs = 0
for key in lmdb_cursor:
    num_imgs += 1
num_batches = int(math.ceil(float(num_imgs) / float(batchsize)))
print 'Num images: %d' % num_imgs
print 'Batch size: %d' % batchsize
print 'Num batches: %d' % num_batches

# Store fc features and labels for all images
fc_feat_dim = embedding_dimension
feat_matrix = np.zeros((num_imgs, fc_feat_dim), dtype=np.float32)
labels = np.zeros((num_imgs, 1), dtype=np.int32)

filename_idx = 0
for batch_id in range(num_batches):
    batch = net.forward()
    fc = net.blobs['fc8_custom'].data.copy()
    l = net.blobs['label'].data.copy()
    num = fc.shape[0]
    for i in range(num):
        if filename_idx + i >= num_imgs:
            break
        feat_matrix[filename_idx+i, :] = fc[i,:]
        labels[filename_idx+i] = l[i]
    filename_idx += num
print 'Extracting features and labels done'

# compute similarity matrix
t = np.ones((num_imgs, 1), dtype=np.float32)
x = np.zeros((num_imgs, 1), dtype=np.float32)
for i in xrange(num_imgs):
    n = np.linalg.norm(feat_matrix[i,:])
    x[i] = n * n
D2 = np.dot(x, t.T) + np.dot(t, x.T) - 2 * np.dot(feat_matrix, feat_matrix.T)
D = np.sqrt(np.absolute(D2))
S = -1 * D
print 'Computing similarity matrix done'

# compute recall at K
num = S.shape[0]
for i in xrange(num):
    S[i,i] = -np.inf;
K = [1, 2, 4, 8, 16, 32]
recall_at_K(labels, S, K)
