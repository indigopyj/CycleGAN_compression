set -ex
python test_RED.py --dataroot ../dataset/ \
--main_G_path checkpoints/viper2cityscapes_cyclegan/45_net_G_A.pth \
--name viper2city_red_iv5 --no_dropout \
--dataset_mode video \
--Viper2Cityscapes \
--phase test \
--num_test 10 \
--layer_idx 13 --max_interval 5