set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="logs/googlenet_contrastive_train.`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./caffe/build/tools/caffe train --gpu $1 \
  --solver models/GoogleNet/solver_contrastive_embed64.prototxt \
  --weights data/imagenet_models/bvlc_googlenet.caffemodel 
