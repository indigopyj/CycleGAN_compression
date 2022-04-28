set -ex
#python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --phase test --no_dropout
python test.py --dataroot ../dataset --name viper2cityscapes_cyclegan --model cycle_gan --phase val --no_dropout --Viper2Cityscapes --serial_batches --epoch 45 --num_test 2 --dataset_mode video
