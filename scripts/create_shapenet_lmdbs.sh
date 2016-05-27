#!/usr/bin/env sh
# Create the shapenet lmdb inputs
# N.B. set the path to the shapenet images

#change this ID and TGT before running

#src should be in DATA/ID and DATA/ID.txt
#tgt should be TGT

#for id in "02747177" "02773838" "02818832" "02828884" "02834778" "02858304" "2871439" "02876657" "02924116" "02933112" "02942699" "02946921" "02954340" "02958343" "02992529" "03046257" "03207941" "03211117" "04074963" "04225987" "04256520" "04330267" "04401088" "04468005" "04530566"
#for id in "03211117" "04401088" "04530566" "04468005" 
for id in "04256520"
do
echo $id

DATA=/

TGT=/cvgl/u/yuxiang/lmdb/"$id"_test_shape/
TXT=/cvgl/u/cwind/Shape_Retrieval/data/"$id"_test_shape.txt

#TGT=/cvgl/u/yuxiang/lmdb/"$id"_test_img/
#TXT=/cvgl/u/cwind/Shape_Retrieval/data/tight_bbx/"$id"_test_img.txt

TOOLS=caffe/build/tools

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$DATA" ]; then
  echo "Error: DATA is not a path to a directory: $DATA"
  echo "Set the DATA variable in create_shapenet.sh to the path" \
       "where the shapenet data is stored."
  exit 1
fi

echo "Creating $ID lmdb..."

#--shuffle if needed.

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    $DATA \
    $TXT \
    $TGT

echo "Done."
done
