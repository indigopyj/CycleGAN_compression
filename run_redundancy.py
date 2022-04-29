"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
from glob import glob
import os

from PIL import Image
import torch
from torch.nn import functional as F
from tqdm import tqdm



from data.base_dataset import get_transform
from models import create_model
from options.test_options import TestOptions


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    transform = get_transform(opt, grayscale=False)

    cc_sum = []
    rmse_sum = [0. for _ in range(11, 20)]

    paths = sorted(glob(os.path.join(opt.dataroot, 'Viper/val/img/*')))
    for path in tqdm(paths, desc='Test       ', position=0, leave=False):
        filenames = sorted(os.listdir(path))[:opt.num_test]

        prev_activations = []
        for i, filename in enumerate(tqdm(filenames, desc='Frame      ', position=1, leave=False)):
            A_img = Image.open(os.path.join(path, filename)).convert("RGB")
            A = transform(A_img).unsqueeze(0).cuda()

            x = A
            st = 0
            activations = []
            for fi in range(11, 20):
                x = model.netG_A.module.model[st:fi](x)
                st = fi
                activations.append(x)

            if i > 0:
                for j, (prev_actv, actv) in enumerate(zip(prev_activations, activations)):
                    rmse = torch.sqrt(F.mse_loss(prev_actv, actv))
                    rmse_sum[j] += rmse.item()
            prev_activations = activations

    for i, fi in enumerate(range(11, 20)):
        print("{}: {}".format(fi - 1, rmse_sum[i] / (len(paths) * opt.num_test)))
