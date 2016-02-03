#!/usr/bin/env sh
# Create the shapenet lmdb inputs
# N.B. set the path to the shapenet images

ID=04004475
EXAMPLE=data
DATA=/capri5/Projects/ImageNet3D/Syn_Images
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

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $DATA/$ID/ \
    $DATA/$ID.txt \
    $EXAMPLE/$ID\_lmdb

echo "Done."
