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
import os

class_id = '04256520'
gpu = 0

MODEL_FILE_IMG = 'models/GoogleNet/' + class_id + '_sobel_img.prototxt'
MODEL_FILE_SHAPE = 'models/GoogleNet/' + class_id + '_sobel_shape.prototxt'
CAFFE_MODEL = '/cvgl/u/jingweij/ObjectNet3D/ImageNet3D/Shape_Retrieval/output/'+class_id+'/googlenet_liftedstructsim_softmax_sobel_embed64_iter_20000.caffemodel'
MEAN_FILE = 'data/imagenet_mean.binaryproto'
#RESULT_IDS = 'data/results/04004475_ids_by_img_3.csv'
TGT = 'data/results/'+class_id+'_sobel/'
#DISTANCE_MATRIX_FILE = TGT+class_id+'_matrix.csv'
DISTANCE_MATRIX_MEAN_FILE = TGT+class_id+'_matrix_mean.csv'
SHAPE_IDS_FILE = TGT+class_id+'_shape_ids.csv'
LMDB_FILENAME_IMG = '/cvgl/u/yuxiang/lmdb/'+class_id+'_test_img/'
LMDB_FILENAME_SHAPE = '/cvgl/u/yuxiang/lmdb/'+class_id+'_test_shape/'

rendered_feature_file = TGT+class_id+'_feature_rendered_imgs.csv'
shape_feature_file = TGT+class_id+'_feature_shape.csv'
test_feature_file = TGT+class_id+'_feature_test.csv'

embedding_dimension = 64
batchsize = 64
if not os.path.exists(TGT):
    os.mkdir(TGT);

caffe.set_device(gpu)
caffe.set_mode_gpu()
net_img = caffe.Net(MODEL_FILE_IMG, CAFFE_MODEL, caffe.TEST)

# open image lmdb
lmdb_env = lmdb.open(LMDB_FILENAME_IMG)
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
feat_matrix_img = np.zeros((num_imgs, fc_feat_dim), dtype=np.float32)
labels_img = np.zeros((num_imgs, 1), dtype=np.int32)

filename_idx = 0
for batch_id in range(num_batches):
    batch = net_img.forward()
    fc = net_img.blobs['fc8_custom'].data.copy()
    l = net_img.blobs['label'].data.copy()
    num = fc.shape[0]
    print '.'
    for i in range(num):
        if filename_idx + i >= num_imgs:
            break
        feat_matrix_img[filename_idx+i, :] = fc[i,:]
        labels_img[filename_idx+i] = l[i]
    filename_idx += num
print 'Extracting features and labels for images done'
print feat_matrix_img.shape
print labels_img.shape

net_shape = caffe.Net(MODEL_FILE_SHAPE, CAFFE_MODEL, caffe.TEST)

# open shape lmdb
lmdb_env = lmdb.open(LMDB_FILENAME_SHAPE)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()

num_shapes = 0
for key in lmdb_cursor:
    num_shapes += 1;
num_batches = int(math.ceil(float(num_shapes)/float(batchsize)));
print 'Num shapes: %d' % num_shapes
print 'Batch size: %d' % batchsize
print 'Num batches: %d' % num_batches

caffe.set_device(gpu)

fc_feat_dim = embedding_dimension
feat_matrix_shape = np.zeros((num_shapes, fc_feat_dim),dtype=np.float32)
labels_shape = np.zeros((num_shapes,1), dtype=np.int32)

filename_idx = 0
for batch_id in range(num_batches):
    batch = net_shape.forward();
    fc = net_shape.blobs['fc8_custom'].data.copy()
    l = net_shape.blobs['label'].data.copy()
    num = fc.shape[0]
    print '.'
    for i in range(num):
	      if filename_idx + i >= num_shapes:
	          break
	      feat_matrix_shape[filename_idx+i,:] = fc[i,:]
	      labels_shape[filename_idx+i] = l[i]
    filename_idx += num
print 'Extracting features and labels for shapes done'
print feat_matrix_shape.shape
print labels_shape.shape

num_shape_classes = np.max(labels_shape) + 1
print num_shape_classes

mean_feat_matrix_shape = np.zeros((num_shape_classes, feat_matrix_shape.shape[1]))
for i in range(num_shape_classes):
    mean_feat_matrix_shape[i,:] = np.mean(feat_matrix_shape[(labels_shape == i)[:,0],:],axis = 0)

# compute distance matrix
distance_matrix = np.zeros((num_imgs, num_shapes))
distance_matrix_mean = np.zeros((num_imgs, num_shape_classes))
for i in range(num_imgs):
    diffs = feat_matrix_shape - feat_matrix_img[i,:]
    diffs = diffs.T;
    distance_matrix[i,:] = np.sum(diffs*diffs, axis=0)

    diffs = mean_feat_matrix_shape - feat_matrix_img[i,:]
    diffs = diffs.T
    distance_matrix_mean[i,:] = np.sum(diffs*diffs,axis=0)

print 'Computing distance matrix done'
print distance_matrix.shape
print distance_matrix_mean.shape
print labels_shape.shape

#np.savetxt(DISTANCE_MATRIX_FILE, distance_matrix, delimiter=',')
np.savetxt(rendered_feature_file, feat_matrix_shape, delimiter=',')
np.savetxt(shape_feature_file, mean_feat_matrix_shape, delimiter=',')
np.savetxt(test_feature_file, feat_matrix_img, delimiter=',')
np.savetxt(DISTANCE_MATRIX_MEAN_FILE, distance_matrix_mean, delimiter=',')
np.savetxt(SHAPE_IDS_FILE, labels_shape, delimiter=',')
