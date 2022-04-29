set -ex
# CUDA_VISIBLE_DEVICES=0,2,3,4 python train_RED.py --dataroot ../dataset/ \
# --main_G_path checkpoints/viper2cityscapes_cyclegan/23_net_G_A.pth \
# --name viper2city_red --pool_size 50 --no_dropout --gpu_ids 0,1,2,3 --batch_size 8 \
# --dataset_mode video \
# --Viper2Cityscapes \
# --layer_idx 13 --max_interval 10 --lambda_L1_out 10 \
# --n_epochs 200 --n_epochs_decay 200 \
# --tensorboard_dir tensorboard/

# CUDA_VISIBLE_DEVICES=5 python train_RED.py --dataroot ../dataset/ \
# --main_G_path checkpoints/viper2cityscapes_cyclegan/40_net_G_A.pth \
# --name viper2city_red_iv5_l1outimg2 --pool_size 50 --no_dropout --gpu_ids 0 --batch_size 8 \
# --dataset_mode video \
# --Viper2Cityscapes \
# --layer_idx 13 --max_interval 5 --lambda_L1_out 10 \
# --n_epochs 500 --n_epochs_decay 200 \
# --tensorboard_dir tensorboard/ \
# --l1_out_img2

CUDA_VISIBLE_DEVICES=0 python train_RED.py --dataroot ../dataset/ \
--main_G_path checkpoints/viper2cityscapes_cyclegan/45_net_G_A.pth \
--name viper2city_red_iv5 --pool_size 50 --no_dropout --gpu_ids 0 --batch_size 8 \
--dataset_mode video \
--Viper2Cityscapes \
--layer_idx 13 --max_interval 5 --lambda_L1_out 10 \
--n_epochs 1000 --n_epochs_decay 200 \
--tensorboard_dir tensorboard/ \
--save_latest_freq 1000

# CUDA_VISIBLE_DEVICES=6,7 python train_RED.py --dataroot ../dataset/ \
# --main_G_path checkpoints/viper2cityscapes_cyclegan/45_net_G_A.pth \
# --main_D_path checkpoints/viper2cityscapes_cyclegan/45_net_D_A.pth \
# --name viper2city_red_iv5_gan_D --pool_size 50 --no_dropout --gpu_ids 0,1 --batch_size 32 \
# --dataset_mode video \
# --Viper2Cityscapes --GAN_loss \
# --layer_idx 13 --max_interval 5 --lambda_L1_out 10 \
# --n_epochs 1000 --n_epochs_decay 200 \
# --tensorboard_dir tensorboard/ --save_latest_freq 500