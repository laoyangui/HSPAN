#!/bin/bash
#Training and testing on scale factor 4.

# Train on DIV2K datasets.
python main.py --dir_data ../../SrTrainingData --data_range 1-800/1-5 --n_GPUs 1 --rgb_range 1 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 1000 --chop --save_results --data_test Set5 --n_resgroups 10 --n_resblocks 4 --n_feats 192 --reduction 2 --topk 128 --res_scale 0.1 --batch_size 16 --model HSPAN --scale 2 --patch_size 96 --save HSPAN_x2 --data_train DIV2K

# Test on Set5, Set14, B100, Urban100, Manga109 datasets.
python main.py --dir_data ../../ --data_test Set5+Set14+B100+Urban100+Manga109 --n_GPUs 1 --rgb_range 1 --save_models --save_results --n_resgroups 10 --n_resblocks 4 --n_feats 192  --reduction 2 --topk 128  --res_scale 0.1 --model HSPAN --save HSPAN_x4_results --chop --data_range 1-800/1-5 --scale 4 --test_only --pre_train ../experiment/HSPAN_x4/model/HSPAN_x4.pt