# nyu 训练执行脚本
# 实验编号 8 
python ./training/train_tum.py --cfg=cfgs/tum/tum_2_2_shufflev2_fast.yaml --dataset_dir=data/tum   --name=nyu_model  --restore=checkpoints/tum/tmu_model/_stage_2.ckpt
