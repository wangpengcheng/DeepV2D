# nyu 训练执行脚本
# 实验1 
python ./training/train_nyuv2.py --cfg=cfgs/nyu/nyu_2_2_resnet_resnet.yaml --name=nyu_model --dataset_dir=data/nyu2 --tfrecords=data/nyu_train/nyu_train.tfrecords --restore=checkpoints/tum/tmu_model/_stage_2.ckpt
