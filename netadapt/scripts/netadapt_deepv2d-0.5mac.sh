CUDA_VISIBLE_DEVICES=0 python master.py models/deepv2d/prune-by-mac 3 224 224 \
    -im models/deepv2d/model.pth.tar -gp 0  \
    -mi 3000 -bur 0.25 -rt FLOPS  -irr 0.025 -rd 0.96 \
    -lr 0.001 -st 500 \
    -dp data/ --arch deepv2d