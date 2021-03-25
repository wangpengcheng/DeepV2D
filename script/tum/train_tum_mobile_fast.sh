# nyu 训练执行脚本
# 实验 2
#python ./training/train_nyuv2.py --cfg=cfgs/nyu/nyu_2_2_mobile_fast.yaml --name=nyu_model --dataset_dir=data/nyu2 --tfrecords=data/nyu/nyu_train.tfrecords --restore=checkpoints/tum/tmu_model/_stage_2.ckpt
python ./training/train_tum.py --cfg=cfgs/tum/tum_2_2_mobile_fast.yaml --name=tmu_model --dataset_dir=data/tum --restore=checkpoints/tum/tmu_model/_stage_2.ckpt
