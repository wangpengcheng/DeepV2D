# nyu 训练执行脚本
# 实验 2
python ./training/train_nyuv2.py --cfg=cfgs/nyu/nyu_2_2_shufflev2_aspp.yaml --name=nyu_model  --tfrecords=data/nyu/nyu_train.tfrecords --restore=checkpoints/tum/tmu_model/_stage_2.ckpt
