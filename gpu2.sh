#!/bin/bash


#CUDA_VISIBLE_DEVICES=1 python generate_adv.py --data=1 --attack=2
#CUDA_VISIBLE_DEVICES=1 python generate_adv.py --data=1 --attack=3
#CUDA_VISIBLE_DEVICES=1 python generate_adv.py --data=1 --attack=4
#CUDA_VISIBLE_DEVICES=1 python generate_adv.py --data=1 --attack=5

#CUDA_VISIBLE_DEVICES=1 python generate_adv.py --data=1 --attack=0 --beam=1
#CUDA_VISIBLE_DEVICES=1 python generate_adv.py --data=1 --attack=0 --beam=2
#CUDA_VISIBLE_DEVICES=1 python generate_adv.py --data=1 --attack=0 --beam=3
#CUDA_VISIBLE_DEVICES=1 python generate_adv.py --data=1 --attack=0 --beam=4
#CUDA_VISIBLE_DEVICES=1 python generate_adv.py --data=1 --attack=0 --beam=5


CUDA_VISIBLE_DEVICES=2 python measure_senstive.py --data=2 --attack=7
