import copy
import os

import torch
from tqdm import tqdm

from data import CustomDatasetDataLoader
from models import networks
from models.base_model import BaseModel
from models.modules.RED_modules import REDNet
from util.util import tensor2im, save_image
import cv2
import numpy as np
from torch.nn import functional as F
from util.image_pool import ImagePool

def init_net(net, gpu_ids):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs

    return net


def create_eval_dataloader(opt, phase="val"):
    opt = copy.deepcopy(opt)
    opt.isTrain = False
    opt.serial_batches = True
    opt.phase = phase # 고쳐야됨
    dataloader = CustomDatasetDataLoader(opt)
    dataloader = dataloader.load_data()
    return dataloader


class REDModel(BaseModel):
    def __init__(self, opt):
        #assert opt.isTrain
        super(REDModel, self).__init__(opt)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['l1', 'l1_out']
        if opt.GAN_loss:
            self.loss_names += ['G_A', 'D_A']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['img1', 'img2', 'fake_diff', 'real_diff']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        self.model_names = ['R']

        self.netR = REDNet(input_nc=256+3+3, ngfs=[256])
        self.netR = init_net(self.netR, opt.gpu_ids)

        # define loss functions
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionL1_out = torch.nn.L1Loss()
        

        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        if opt.isTrain:
            self.optimizer = torch.optim.Adam(self.netR.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer)
            
            if opt.GAN_loss:
                self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
                self.fake_B_pool = ImagePool(opt.pool_size)
        self.eval_dataloader = create_eval_dataloader(self.opt, "val")
        

    def setup_with_G(self, opt, model, verbose=True):
        self.modelG = model.netG_A
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if verbose:
            self.print_networks(verbose)
        for param in self.modelG.parameters():
            param.requires_grad = False
        print("freeze main generator")
    
    def setup_with_D(self, opt, model, verbose=True):
        self.modelD = model.netD_A
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if verbose:
            self.print_networks(verbose)
        for param in self.modelD.parameters():
            param.requires_grad = False
        print("freeze main discriminator")

    def set_input(self, input):
        self.img1 = input['img1'].to(self.device)
        self.img2 = input['img2'].to(self.device)
        self.img1_paths = input['img1_paths']
        self.image_root = input['img_root']

    
    def set_test_input(self, img_paths):
        from PIL import Image
        from data.base_dataset import get_transform
        self.transform = get_transform(self.opt, grayscale=False)
        self.next_img_paths = img_paths
        self.next_img = []
        for i in range(len(img_paths)):
            next_img = Image.open(img_paths[i]).convert("RGB")
            next_img = self.transform(next_img).unsqueeze(0)
            self.next_img.append(next_img)
        self.next_img = torch.cat(self.next_img, dim=0).to(self.device)

    def forward(self):
        b = self.img1.size(0)
        activations = self.modelG.module.model[:self.opt.layer_idx](torch.cat((self.img1, self.img2), 0)).detach()  # TODO: hyper-parameter tuning
        self.real_diff = activations[b:] - activations[:b]
        if self.opt.crop_size == 256:
            resize_size = 64
        elif self.opt.crop_size == 512:
            resize_size = 128
        img1_resized = F.interpolate(self.img1, size=resize_size, mode='bicubic')
        img2_resized = F.interpolate(self.img2, size=resize_size, mode='bicubic')
        self.fake_diff = self.netR(torch.cat((img1_resized, img2_resized, activations[:b]), 1), 0)
        self.real_im = self.modelG.module.model[self.opt.layer_idx:](activations[b:]).detach()
        self.fake_im = self.modelG.module.model[self.opt.layer_idx:](activations[:b] + self.fake_diff)

    def backward(self):
        lambda_l1 = self.opt.lambda_L1
        lambda_l1_out = self.opt.lambda_L1_out
        self.loss_l1 = self.criterionL1(self.fake_diff, self.real_diff)
        self.loss_l1_out = self.criterionL1_out(self.fake_im, self.real_im)
        
        self.loss = self.loss_l1 * lambda_l1 + self.loss_l1_out * lambda_l1_out
        
        # GAN loss
        if self.opt.GAN_loss:
            self.loss_G_A = self.criterionGAN(self.modelD(self.fake_im), True)
            self.loss += self.loss_G_A
        self.loss.backward()
        
        
    # def backward_D_basic(self, netD, real, fake):
    #     # Real
    #     pred_real = netD(real)
    #     loss_D_real = self.criterionGAN(pred_real, True)
    #     # Fake
    #     pred_fake = netD(fake.detach())
    #     loss_D_fake = self.criterionGAN(pred_fake, False)
    #     # Combined loss and calculate gradients
    #     loss_D = (loss_D_real + loss_D_fake) * 0.5
    #     loss_D.backward()
    #     return loss_D
    
    # def backward_D_A(self):
    #     """Calculate GAN loss for discriminator D_A"""
    #     fake_B = self.fake_B_pool.query(self.fake_im)
    #     self.loss_D_A = self.backward_D_basic(self.modelD, self.real_im, fake_B)

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def evaluate_model(self, step):
        self.save_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        save_dir = os.path.join(self.save_dir, 'eval', str(step))
        os.makedirs(save_dir, exist_ok=True)
        self.netR.eval()
        
        
        with torch.no_grad(): 
            for i, data_i in enumerate(tqdm(self.eval_dataloader, desc='Eval       ', position=2, leave=False)):
                self.set_input(data_i)
                activations = self.modelG.module.model[:self.opt.layer_idx](self.img1) # randomly chosen image
                
                for j in range(1, self.opt.max_interval):
                    # self.img1 : reference frame(past frame)
                    # self.next_img : next image
                    img2_paths = []
                    for batch_idx in range(len(self.img1_paths)):
                        img1_name, img1_ext = os.path.splitext(self.img1_paths[batch_idx])
                        img2_idx = int(img1_name.split("_")[1]) + j
                        img2_name =  "%s_%05d%s" %(img1_name.split("_")[0], img2_idx, img1_ext)
                        img2_path = os.path.join(self.image_root[batch_idx], img2_name) 
                        img2_paths.append(img2_path)
                    self.set_test_input(img2_paths) # load an next image (img1 + interval) as a batch
                    img1_resized = F.interpolate(self.img1, size=activations.shape[2:], mode='bicubic')
                    nextimg_resized = F.interpolate(self.next_img, size=activations.shape[2:], mode='bicubic')
                    fake_diff = self.netR(torch.cat((img1_resized, nextimg_resized, activations), 1), 0)
                    real_im = self.modelG.module.model(self.next_img)
                    fake_im = self.modelG.module.model[self.opt.layer_idx:](activations + fake_diff)
            
                    for k in range(len(self.img1_paths)):
                        img1_name, _= os.path.splitext(self.img1_paths[k])
                        name = f"{img1_name}_{j}.png" # interval_originalname
                        input1_im = tensor2im(self.img1, idx=k)
                        input2_im = tensor2im(self.next_img, idx=k)
                        real = tensor2im(real_im, idx=k)
                        fake = tensor2im(fake_im, idx=k)
                        save_image(input1_im, os.path.join(save_dir, 'input1', '%s' % self.img1_paths[k]), create_dir=True)
                        save_image(input2_im, os.path.join(save_dir, 'input2', '%s' % name), create_dir=True)
                        save_image(real, os.path.join(save_dir, 'real', '%s' % name), create_dir=True)
                        save_image(fake, os.path.join(save_dir, 'fake', '%s' % name), create_dir=True)

        self.netR.train()
    
    def test_model(self, result_path):
        
        self.save_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        
        self.netR.eval()
        self.test_dataloader = create_eval_dataloader(self.opt, "test")
        if self.opt.crop_size == 256:
            resize_size = 64
        elif self.opt.crop_size == 512:
            resize_size = 128

        with torch.no_grad():
            for seq_idx, seq_i in enumerate(tqdm(self.test_dataloader, desc='Eval       ', position=2, leave=False)):
                if seq_idx >= self.opt.num_test:  # only apply our model to opt.num_test videos.
                    break
               
                vid_name = seq_i['seq_path'][0].split("/")[-1]
                video = cv2.VideoWriter(os.path.join(result_path, vid_name+'.mp4'), fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=10, frameSize=(768, 256))
                img_list = sorted(os.listdir(seq_i['seq_path'][0]))[:400] # limit length of test video
                for i, data_i in enumerate(img_list):
                    data_path = os.path.join(seq_i['seq_path'][0], data_i)
                    data_path = [data_path]
                    self.set_test_input(data_path)
                    if i % self.opt.max_interval == 0:
                        reference_img = self.next_img
                        fake_im = self.modelG.module.model(self.next_img)
                        real_im = fake_im
                        activations = self.modelG.module.model[:self.opt.layer_idx](reference_img)
                        ref_resized = F.interpolate(reference_img, size=resize_size, mode='bicubic')
                    else:
                        
                        nextimg_resized = F.interpolate(self.next_img, size=resize_size, mode='bicubic')
                        fake_diff = self.netR(torch.cat((ref_resized, nextimg_resized, activations), 1), 0)
                        real_im = self.modelG.module.model(self.next_img)
                        fake_im = self.modelG.module.model[self.opt.layer_idx:](activations + fake_diff)
                    
                    
                    name = f"{vid_name}_{i}.png"
                    input_im = tensor2im(self.next_img)
                    real_im = tensor2im(real_im)
                    fake_im = tensor2im(fake_im)
                    cat_img = np.concatenate((input_im, real_im, fake_im), axis=1)
                    save_image(input_im, os.path.join(result_path, 'input', '%s' % name), create_dir=True)
                    save_image(real_im, os.path.join(result_path, 'real', '%s' % name), create_dir=True)
                    save_image(fake_im, os.path.join(result_path, 'fake', '%s' % name), create_dir=True)
                    cat_img = cv2.cvtColor(cat_img, cv2.COLOR_RGB2BGR)
                    video.write(cat_img)
                    