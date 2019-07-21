#!/bin/bash

detach=false

#at=self
#
#lv1=1
#lv2=2
#tag=l${lv1}${lv2}_d${detach}_at${at}
#echo ${tag}
#rm -rf tensorboard/$tag
#python resattvae_train.py --tag $tag --detach ${detach} --lv1 ${lv1} --lv2 ${lv2} --attentiontype ${at}
#
#lv1=3
#lv2=4
#tag=l${lv1}${lv2}_d${detach}_at${at}
#echo ${tag}
#rm -rf tensorboard/$tag
#python resattvae_train.py --tag $tag --detach ${detach} --lv1 ${lv1} --lv2 ${lv2} --attentiontype ${at}

at=SE

lv1=1
lv2=2
tag=l${lv1}${lv2}_d${detach}_at${at}
echo ${tag}
rm -rf tensorboard/$tag
python resattvae_train.py --tag $tag --detach ${detach} --lv1 ${lv1} --lv2 ${lv2} --attentiontype ${at} --batch_size 10000

lv1=3
lv2=4
tag=l${lv1}${lv2}_d${detach}_at${at}
echo ${tag}
rm -rf tensorboard/$tag
python resattvae_train.py --tag $tag --detach ${detach} --lv1 ${lv1} --lv2 ${lv2} --attentiontype ${at} --batch_size 10000
