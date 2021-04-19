# nyu 训练执行脚本
# 实验编号 8 
python ./training/train_nyuv2.py --cfg=cfgs/nyu/nyu_2_2_shuffle_fast.yaml --name=nyu_model --tfrecords=data/nyu/nyu_train.tfrecords --restore=checkpoints/tum/tmu_model/_stage_2.ckpt
