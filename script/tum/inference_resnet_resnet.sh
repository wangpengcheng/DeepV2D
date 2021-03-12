# nyu推理测试脚本

python ./demos/demo_v2d.py --cfg=cfgs/nyu/nyu_2_2_resnet_resnet.yaml  --model=checkpoints/nyu/resnet_resnet/_stage_2.ckpt --use_pose --sequence=data/demos/bathroom_0001 --inference_file_name=data/demos/bathroom_0001/rgb_depth_ground.txt


# nyu 训练执行脚本
# 实验1 
# python ./training/train_nyuv2.py --cfg=cfgs/nyu/nyu_2_2_resnet_resnet.yaml --name=nyu_model --tfrecords=data/nyu_train/nyu_train.tfrecords --restore=checkpoints/tum/tmu_model/_stage_2.ckpt
