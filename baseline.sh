#!/bin/bash 

tag=baseline
echo ${tag}
rm -rf tensorboard/$tag
python resattvae_train.py --tag $tag  --model "baseline"

detach=false
lv1=1
lv2=3
tag=l${lv1}${lv2}_d${detach}
echo ${tag}
rm -rf tensorboard/$tag
python resattvae_train.py --tag $tag --detach ${detach} --lv1 ${lv1} --lv2 ${lv2}
