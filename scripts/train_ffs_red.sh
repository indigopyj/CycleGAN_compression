set -ex
CUDA_VISIBLE_DEVICES=2,3,4,5 python train_RED.py --dataroot ../dataset/random_FFS/ \
--main_G_path checkpoints/ffs_cyclegan/200_net_G_A.pth \
--name ffs_red --pool_size 50 --no_dropout --gpu_ids 0,1,2,3 --batch_size 8 \
--dataset_mode video \
--serial_batches \
--tensorboard_dir tensorboard/
