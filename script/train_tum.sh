#python ./training/train_tum.py --cfg=cfgs/tum.yaml --name=tmu_model --dataset_dir=data/tum --restore=checkpoints/tum/tmu_model/_stage_2.ckpt 
#python ./training/train_tum.py --cfg=cfgs/tum.yaml --name=tmu_model --dataset_dir=data/tum  --restore=models/nyu.ckpt 
export CUDA_VISIBLE_DEVICES=1
#python ./training/train_tum.py --cfg=cfgs/tum.yaml --name=tmu_model --dataset_dir=data/tum  --restore=models/nyu.ckpt 
python ./training/train_tum.py --cfg=cfgs/tum.yaml --name=tmu_model --dataset_dir=data/tum --restore=checkpoints/tum/tmu_model_240_320/_stage_2.ckpt 