import argparse
import os
import random

import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm


from cyclegan_pytorch import ImageDataset
from model import CGAN_model
dataroot = "./data"
dataset_p = "prof2drawing"
image_size = 256
outf_p = "./results"
cuda = True


try:
    os.makedirs(outf_p)
except OSError:
    pass

manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Dataset
dataset = ImageDataset(root=os.path.join(dataroot, dataset_p),
                       transform=transforms.Compose([
                           transforms.Resize(image_size),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                       ]),
                       mode="test")

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

try:
    os.makedirs(os.path.join(outf_p, str(dataset_p), "A"))
    os.makedirs(os.path.join(outf_p, str(dataset_p), "B"))
except OSError:
    pass

device = torch.device("cuda:0" if cuda else "cpu")

# create model

cgan_model = CGAN_model(100, 500, 0.0002, 1)
netG_A2B = cgan_model.netG_A2B
netG_B2A = cgan_model.netG_B2A


# Load state dicts

netG_A2B.load_state_dict(torch.load("weights/prof2drawing/netG_A2B.pth"))
netG_B2A.load_state_dict(torch.load("weights/prof2drawing/netG_B2A.pth"))

# Set model mode
netG_A2B.eval()
netG_B2A.eval()

progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

for i, data in progress_bar:
    # get batch size data
    real_images_A = data["A"].to(device)
    real_images_B = data["B"].to(device)

    # Generate output
    fake_image_A = 0.5 * (cgan_model.netG_B2A(real_images_B).data + 1.0)
    fake_image_B = 0.5 * (cgan_model.netG_A2B(real_images_A).data + 1.0)

    # Save image files
    vutils.save_image(fake_image_A.detach(), f"{outf_p}/{dataset_p}/A/{i + 1:04d}.png", normalize=True)
    vutils.save_image(fake_image_B.detach(), f"{outf_p}/{dataset_p}/B/{i + 1:04d}.png", normalize=True)

    progress_bar.set_description(f"Process images {i + 1} of {len(dataloader)}")
