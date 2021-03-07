#python ./training/train_tum.py --cfg=cfgs/tum_resnet.yaml --name=tmu_model --dataset_dir=data/tum --restore=checkpoints/tum/tmu_model01/_stage_2.ckpt 
#python ./training/train_tum.py --cfg=cfgs/tum.yaml --name=tmu_model --dataset_dir=data/tum --restore=checkpoints/tum/tmu_model/_stage_2.ckpt 
#python ./training/train_tum.py --cfg=cfgs/tum.yaml --name=tmu_model --dataset_dir=data/tum  --restore=models/nyu.ckpt 

#python ./training/train_tum.py --cfg=cfgs/tum.yaml --name=tmu_model --dataset_dir=data/tum  --restore=models/nyu.ckpt 
#python ./training/train_tum.py --cfg=cfgs/tum.yaml --name=tmu_model --dataset_dir=data/tum --restore=checkpoints/tum/tmu_model_240_320/_stage_2.ckpt 

#python ./training/train_tum.py --cfg=cfgs/tum_2_2_fast.yaml --name=tmu_model --dataset_dir=data/tum --restore=checkpoints/tum/tmu_model_240_320_2_2/_stage_2.ckpt 


#python ./training/train_tum.py --cfg=cfgs/tum_2_2_moblie.yaml --name=tmu_model --dataset_dir=data/tum --restore=checkpoints/tum/tmu_model_240_320_2_2/_stage_2.ckpt 
python ./training/train_nyuv2.py --cfg=cfgs/nyu_2_2_fast_shufflev2.yaml  --dataset_dir=data/tum --name=nyu_model --dataset_dir=data/nyu2 --tfrecords=data/nyu/nyu_train.tfrecords --restore=checkpoints/tum/tmu_model/_stage_2.ckpt
#python ./training/train_tum.py --cfg=cfgs/tum_2_2_moblie.yaml --name=tmu_model --dataset_dir=data/tum --restore=checkpoints/tum/tmu_model/_stage_2.ckpt

