#!/usr/bin/python
# coding: utf-8

import os
import sys
from shutil import copyfile
import numpy as np

K=20;
class_id = '04256520';

shape_example_imgs = '/cvgl/u/cwind/KDE/RenderForCNN-master/data/syn_images/'+class_id+'/';
#shape_imgs = '/cvgl/u/cwind/Shape_Retrieval/data/'+class_id+'_test_shape/'
shape_imgs_list = '/cvgl/u/cwind/Shape_Retrieval/data/'+class_id+'_test_shape.txt'
#test_imgs = '/cvgl/u/cwind/Shape_Retrieval/data/'+class_id+'_test_img/'
test_imgs_list = '/cvgl/u/cwind/Shape_Retrieval/data/tight_bbx/'+class_id+'_test_img.txt'
distance_matrix_file = 'data/results/'+class_id+'_sobel/'+class_id+'_matrix_mean.csv';
#shape_imgs_labels = '/cvgl/u/cwind/Shape_Retrieval/data/results/'+class_id+'/'+class_id+'_shape_ids.csv';

#rendered_feature_file = '/cvgl/u/cwind/Shape_Retrieval/data/results/'+class_id+'/'+class_id+'_feature_rendered_imgs.csv';
#shape_feature_file = '/cvgl/u/cwind/Shape_Retrieval/data/results/'+class_id+'/'+class_id+'_feature_shape.csv';
#test_feature_file = '/cvgl/u/cwind/Shape_Retrieval/data/results/'+class_id+'/'+class_id+'_feature_test.csv';

results = '/cvgl/group/ImageNet3D/Retrievals/'+class_id+'_sobel/';

if not os.path.exists(results):
    os.mkdir(results);

shape_id_map = {};
img_id_map = {};
index = 0;
for line in open(shape_imgs_list,'r'):
    shape_id_map[int(line.split(' ')[1])]=line.split(' ')[0];
    index = index+1;

index = 0;
for line in open(test_imgs_list,'r'):
    img_id_map[index]=line.split(' ')[0];
    index = index+1;
print 'Loading shape images list done';


distance_matrix = np.loadtxt(open(distance_matrix_file), delimiter=',');
#shape_labels = np.loadtxt(open(shape_imgs_labels),delimiter=',');
print 'Loading distance matrix done';
print distance_matrix.shape;

ids = np.zeros(distance_matrix.shape);
for i in range(distance_matrix.shape[0]):
    ids[i,:] = np.argsort(distance_matrix[i,:]);

ids = ids[:,:K];
for index in range(ids.shape[0]):
    img = img_id_map[index];
    img_name = img.split('/')[-1];
    path = results+img_name;
    #src_path = test_imgs+img;
    src_path = img;
    if not os.path.exists(path):
	os.makedirs(path);
    shape_list_file = open(path+'/list.txt', 'w');
    copyfile(src_path,path+'/'+img_name);
    for i in range(ids.shape[1]):
	shape_img_name = None;
	shape_name = shape_id_map[int(ids[index,i])].split('/')[-1].split('_')[1];
	src_path = shape_example_imgs+shape_name;
	shape_list_file.write(shape_name+'\n');
	#print src_path
	for n in os.listdir(src_path):
	    shape_img_name = n;
	    break;
	#print int(ids[index,i]);
  	#print src_path
        copyfile(src_path+'/'+shape_img_name,'%s/%d.png'%(path,i));
    #break;
    
