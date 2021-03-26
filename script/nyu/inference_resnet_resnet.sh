# nyu推理测试脚本
# 主要是进行推理和测试时间
python3 ./demos/demo_v2d_origin.py --cfg=cfgs/nyu/nyu_2_2_shufflev2_fast.yaml  --model=checkpoints/nyu/shufflenetv2_fast/_stage_2.ckpt --use_pose --sequence=data/nyu2/bedroom_0129 --inference_file_name=data/nyu2/bedroom_0129/rgb_depth_ground.txt

#./demos/demo_v2d.py
# nyu 训练执行脚本
# 实验1 
# python ./training/train_nyuv2.py --cfg=cfgs/nyu/nyu_2_2_resnet_resnet.yaml --name=nyu_model --tfrecords=data/nyu_train/nyu_train.tfrecords --restore=checkpoints/tum/tmu_model/_stage_2.ckpt
