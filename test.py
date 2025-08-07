import argparse
import sys
import time
import numpy as np
import pyiqa
import torchvision
import itertools
from data.metrics import psnr,ssim
import warnings
import torch
import torch.backends.cudnn as cudnn
from torch import optim, nn
from torch.utils.data import DataLoader
from data import utils
from data.dataloader import TrainDataloader,TestDataloader
from networks import DCP,RJNet,RTNet,PF,GF
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
from losses import cap_loss,VGG19CR

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--hazy_val_dir', type=str, default='./datasets/reals/', help='')
parser.add_argument('--clear_val_dir', type=str, default='./datasets/reals/', help='')
parser.add_argument('--save_img_dir', default='./results/',help='Directory to save the model')
parser.add_argument('--batch_size', type=int, default=1)
args = parser.parse_args('')

if __name__ == "__main__":

    transforms_train = [
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    val_dataset = TestDataloader(args.hazy_val_dir, args.hazy_val_dir, transform=transforms_train)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)

    dataset_length_val = len(val_loader)

    DCP = DCP.DCPDehazeGenerator().to(device)  # No learnable parameters
    Refined_T = GF.Refined_T().to(device)  # No learnable parameters

    JNet = RJNet.JNet().to(device)
    TNet = RTNet.TNet().to(device)
    print('The networks are instantiated successfully！')

    total_params = sum(p.numel() for p in JNet.parameters() if p.requires_grad) + sum(p.numel() for p in TNet.parameters() if p.requires_grad)
    print("Total_params: ==> {}".format(total_params))

    JNet.load_state_dict(torch.load("checkpoints/RW_JNet.pth"), False)
    TNet.load_state_dict(torch.load("checkpoints/RW_TNet.pth"), False)
    print('Model weights loaded successfully！')

    test_time = 0

    musiq=[]
    brisque=[]
    niqe=[]

    musiq_metric = pyiqa.create_metric('musiq').cuda()
    niqe_metric = pyiqa.create_metric('niqe').cuda()

    with torch.no_grad():

        JNet.eval()
        TNet.eval()

        torch.cuda.empty_cache()

        start_time = time.time()

        for a, batch_val in enumerate(val_loader):

            haze = batch_val[0].to(device)
            # clear = batch_val[1].to(device)

            image_name = batch_val[2][0]

            J_DCP_Val, T_DCP_Val, A_DCP_Val = DCP(haze)

            T_Coarse_Val = TNet(torch.cat([T_DCP_Val, haze], dim=1))

            T_Refined_Val = Refined_T(haze, T_Coarse_Val)

            J_Refined_Val, _ = JNet(J_DCP_Val)

            J_ASM_Val = utils.reverse_fog_asm(haze, T_Refined_Val, A_DCP_Val)

            J_Fusion_Val = PF.Perceptual_Fusion(haze, J_ASM_Val, J_Refined_Val).to(device)

            torchvision.utils.save_image(torch.cat([haze, J_Fusion_Val], dim=0), args.save_img_dir + "{}".format(image_name))

            musiq.append(musiq_metric(J_Fusion_Val))
            niqe.append(niqe_metric(J_Fusion_Val))

        test_time = (time.time() - start_time) / dataset_length_val

        avg_musiq = sum(musiq) / dataset_length_val
        avg_niqe = sum(niqe) / dataset_length_val

        print("Total test sample processing completed!")
        print("avg_niqe：{:.3f}".format(avg_niqe.item()))
        print("avg_musiq：{:.3f}".format(avg_musiq.item()))
        print("Testing time：{} s".format(test_time))














