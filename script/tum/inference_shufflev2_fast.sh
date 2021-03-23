# nyu推理测试脚本

python ./demos/demo_v2d_origin.py --cfg=cfgs/tum/tum_2_2_shufflev2_fast.yaml  --model=checkpoints/tum/shufflenetv2_fast/_stage_2.ckpt --use_pose --sequence=data/tum/rgbd_dataset_freiburg3_cabinet --inference_file_name=data/tum/rgbd_dataset_freiburg3_cabinet/rgb_depth_ground.txt


# nyu 训练执行脚本
# 实验1 
# python ./training/train_nyuv2.py --cfg=cfgs/nyu/nyu_2_2_resnet_resnet.yaml --name=nyu_model --tfrecords=data/nyu_train/nyu_train.tfrecords --restore=checkpoints/tum/tmu_model/_stage_2.ckpt
