# tum推理测试脚本

python ./demos/demo_v2d.py --cfg=cfgs/tum_2_2.yaml --model=checkpoints/tum/tmu_model_240_320_2_2/_stage_2.ckpt --use_pose --sequence=data/tum/cabinet --inference_file_name=data/tum/cabinet/rgb_depth_ground_test.txt

#python ./demos/demo_v2d.py --cfg=cfgs/tum_2_2_fast.yaml --model=checkpoints/tum/tmu_model/_stage_2.ckpt --use_pose --sequence=data/tum/cabinet --inference_file_name=data/tum/cabinet/rgb_depth_ground_test.txt