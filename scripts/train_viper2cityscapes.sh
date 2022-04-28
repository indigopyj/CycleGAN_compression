set -ex
CUDA_VISIBLE_DEVICES=1,2,3,4 python train.py --dataroot /home/nas2_userH/yeojeongpark/compression/dataset \
--name viper2cityscapes_cyclegan \
--model cycle_gan --pool_size 50 --no_dropout --batch_size 8 --gpu_ids 0,1,2,3 \
--display_server http://143.248.159.145 --display_port 10000 --Viper2Cityscapes