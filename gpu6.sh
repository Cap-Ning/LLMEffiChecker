#!/bin/bash




#CUDA_VISIBLE_DEVICES=5 python generate_adv.py --data=3 --attack=3
#CUDA_VISIBLE_DEVICES=5 python generate_adv.py --data=3 --attack=4
#CUDA_VISIBLE_DEVICES=5 python generate_adv.py --data=3 --attack=5

#CUDA_VISIBLE_DEVICES=6 python generate_adv.py --data=2 --attack=6 --beam=1
#CUDA_VISIBLE_DEVICES=6 python generate_adv.py --data=2 --attack=6 --beam=2
#CUDA_VISIBLE_DEVICES=6 python generate_adv.py --data=2 --attack=6 --beam=3
#CUDA_VISIBLE_DEVICES=6 python generate_adv.py --data=2 --attack=6 --beam=4
#CUDA_VISIBLE_DEVICES=6 python generate_adv.py --data=2 --attack=6 --beam=5


# CUDA_VISIBLE_DEVICES=6 python measure_senstive.py --data=2 --attack=6

CUDA_VISIBLE_DEVICES=6 python measure_latency.py --data=2 --attack=7
