set -ex
CUDA_VISIBLE_DEVICES=4,5 python train.py --dataroot ../dataset/random_FFS/ --name ffs_cyclegan --model cycle_gan --pool_size 50 --no_dropout --display_server http://143.248.159.126 --display_port 10000 --gpu_ids 0,1 --batch_size 4
