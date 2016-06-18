import os
import sys
import shutil
from subprocess import call
import math
import argparse
import datetime

parser = argparse.ArgumentParser(description='Input the shape id to be processed.')
parser.add_argument('-n', type=str, help='Input the shape id.')
# parser.add_argument('-t', type=str, default = '20000')
# parser.add_argument('-gpu', type=int)
args = parser.parse_args()

specified_cls = args.n 

images_root_folder = '/cvgl/group/ImageNet3D/Retrievals/'
# class_ids = os.listdir(images_root_folder)
# class_ids.sort()
class_ids = [specified_cls]
print class_ids

dst_root_folder = '/cvgl/group/ImageNet3D/Retrievals_Aligned_sobel/'
if not os.path.exists(dst_root_folder):
    os.mkdir(dst_root_folder)
g_blender_executable_path = '/cvgl/u/cwind/JointEmbedding/3rd_party/blender/blender'
g_blank_blend_file_path = '/cvgl/u/jingweij/ObjectNet3D/ImageNet3D/Shape_Retrieval/scripts/blank.blend'
render_model_views_path = '/cvgl/u/jingweij/ObjectNet3D/ImageNet3D/Shape_Retrieval/scripts/render_model_views.py'
shapenetcore_root_folder = '/cvgl/group/ShapeNet/ShapeNetCore.v1/'
view_point_root_folder = '/cvgl/group/ImageNet3D/Labels_new'
distance_param = '1.33'

for i, cls in enumerate(class_ids):
#     if i != 0: continue
    print '%d/%d %s' % (i+1, len(class_ids), cls)
    img_subfolder_list = os.listdir(os.path.join(images_root_folder, cls+'_sobel'))
    if not os.path.exists(os.path.join(dst_root_folder, cls)):
        os.mkdir(os.path.join(dst_root_folder, cls))
    length_img_subfolder = len(img_subfolder_list)

    for j, img_sfd in enumerate(img_subfolder_list):
        # if j % 50 == 0 or j == length_img_subfolder-1:
        print "[%s] %d/%d" % (datetime.datetime.now(), j, length_img_subfolder)

        if j > 10: continue
        
        img_name = img_sfd.split('.')[0]
        annotation_order = int(img_name.split('_')[2])
        img_name = '_'.join(img_name.split('_')[0:2])
        
        # extract the view point parameters 
        label_txt = os.path.join(view_point_root_folder, img_name+'.txt')
        with open(label_txt) as label_file:
            annotation = None
            count = 1
            for line in label_file:
                if count == annotation_order:
                    annotation = line
                    break
                count += 1
        if annotation == None: continue
        words = annotation.split()
        try:
        	len(words) == 5 or len(words) == 8
        except:
            print 'Wrong label format! %s' % label_txt
        if len(words) == 8:
            viewpoints = words[5:8]
            conversed = []
            for param in viewpoints:
                conversed.append(str(float(param) * 180 / math.pi))
            viewpoints = conversed
            viewpoints.append(str(distance_param))
        else:
            continue
        
        # create destination folder for this image
        img_to_be_processed = os.path.join(images_root_folder, cls+'_sobel', img_sfd, img_sfd)
        dst_folder = os.path.join(dst_root_folder, cls, img_sfd)
        if not os.path.exists(dst_folder):
            os.mkdir(dst_folder)

        # write view point parameters into a txt in destination folder
        view_point_txt = os.path.join(dst_folder, img_sfd.split('.')[0]+'.txt')
        with open(view_point_txt, 'w') as view_point_file:
            view_point_file.write(' '.join(viewpoints))

        # copy the query image into destination folder
        shutil.copy(img_to_be_processed, dst_folder)
        
        # Read in the retrieved shape md5 list.
        shape_list_txt = os.path.join(images_root_folder, cls+'_sobel', img_sfd, 'list.txt')
        shape_list = []
        with open(shape_list_txt) as shape_list_file:
            for line in shape_list_file:
                line = line.split()
                shape_list.append(line[0])
        
        # Copy the shape list text into destination folder
        shutil.copy(shape_list_txt, dst_folder)
        
        # Render images
        for md5 in shape_list:
            command = '%s %s --background --python %s -- %s %s %s %s %s > /dev/null 2>&1' % \
                (g_blender_executable_path, g_blank_blend_file_path, render_model_views_path, \
                os.path.join(shapenetcore_root_folder, cls, md5, 'model.obj'), \
                cls, md5, view_point_txt, dst_folder);

#             print command
            os.system(command);
        
        # Rename rendered images
        for k in xrange(len(shape_list)):
            for filename in os.listdir(dst_folder):
                if filename.startswith(cls+'_'+shape_list[k]):
                    os.rename(os.path.join(dst_folder, filename), os.path.join(dst_folder, str(k)+'.png'))
print class_ids
