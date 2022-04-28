set -ex
python test.py --dataroot ../dataset/random_FFS/ --name ffs_cyclegan --model cycle_gan --phase test --no_dropout
