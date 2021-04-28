import argparse
import itertools
import os
import random
from cyclegan_pytorch import ImageDataset
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import CGAN_model

dataroot = "./data"
dataset_p = "trump2biden"
epochs = 3
decay_epochs = 100
batch_size = 1
lr = 0.0002
print_freq = 100
image_size = 256
netG_A2B_p = ''
netG_B2A_p = ''
netD_A_p = ''
netD_B_p = ''
outf_p = "./outputs"
cuda = True

try:
    os.makedirs(outf_p)
except OSError:
    pass

try:
    os.makedirs("weights")
except OSError:
    pass

# if args.manualSeed is None:
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
                           transforms.Resize(int(image_size * 1.12), Image.BICUBIC),
                           transforms.RandomCrop(image_size),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                       unaligned=True)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

try:
    os.makedirs(os.path.join(outf_p, dataset_p, "A"))
    os.makedirs(os.path.join(outf_p, dataset_p, "B"))
except OSError:
    pass

try:
    os.makedirs(os.path.join("weights", dataset_p))
except OSError:
    pass

cgan_model = CGAN_model(decay_epochs, epochs, lr, len(dataloader)-1)

for epoch in range(0, epochs):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, data in progress_bar:
        # get batch size data
        device = torch.device("cuda:0")
        real_image_A = data["A"].to(device)
        real_image_B = data["B"].to(device)
        batch_size = real_image_A.size(0)

        errD_A, errD_B, errG, loss_identity_A, loss_identity_B, loss_GAN_A2B, loss_GAN_B2A, loss_cycle_ABA, loss_cycle_BAB = cgan_model.train_data(
            real_image_A, real_image_B, batch_size)

        progress_bar.set_description(
            f"[{epoch}/{epochs - 1}][{i}/{len(dataloader) - 1}] "
            f"Loss_D: {(errD_A + errD_B).item():.4f} "
            f"Loss_G: {errG.item():.4f} "
            f"Loss_G_identity: {(loss_identity_A + loss_identity_B).item():.4f} "
            f"loss_G_GAN: {(loss_GAN_A2B + loss_GAN_B2A).item():.4f} "
            f"loss_G_cycle: {(loss_cycle_ABA + loss_cycle_BAB).item():.4f}")

        if i % print_freq == 0:
            vutils.save_image(real_image_A,
                              f"{outf_p}/{dataset_p}/A/real_samples.png",
                              normalize=True)
            vutils.save_image(real_image_B,
                              f"{outf_p}/{dataset_p}/B/real_samples.png",
                              normalize=True)

            fake_image_A = 0.5 * (cgan_model.netG_B2A(real_image_B).data + 1.0)
            fake_image_B = 0.5 * (cgan_model.netG_A2B(real_image_A).data + 1.0)

            vutils.save_image(fake_image_A.detach(),
                              f"{outf_p}/{dataset_p}/A/fake_samples_epoch_{epoch}.png",
                              normalize=True)
            vutils.save_image(fake_image_B.detach(),
                              f"{outf_p}/{dataset_p}/B/fake_samples_epoch_{epoch}.png",
                              normalize=True)

    # do check pointing
    torch.save(cgan_model.netG_A2B.state_dict(), f"weights/{dataset_p}/netG_A2B_epoch_{epoch}.pth")
    torch.save(cgan_model.netG_B2A.state_dict(), f"weights/{dataset_p}/netG_B2A_epoch_{epoch}.pth")
    torch.save(cgan_model.netD_A.state_dict(), f"weights/{dataset_p}/netD_A_epoch_{epoch}.pth")
    torch.save(cgan_model.netD_B.state_dict(), f"weights/{dataset_p}/netD_B_epoch_{epoch}.pth")

    cgan_model.Update_lr(epoch, epochs, decay_epochs)

cgan_model.plot_losses()


# save last check pointing
torch.save(cgan_model.netG_A2B.state_dict(), f"weights/{dataset_p}/netG_A2B.pth")
torch.save(cgan_model.netG_B2A.state_dict(), f"weights/{dataset_p}/netG_B2A.pth")
torch.save(cgan_model.netD_A.state_dict(), f"weights/{dataset_p}/netD_A.pth")
torch.save(cgan_model.netD_B.state_dict(), f"weights/{dataset_p}/netD_B.pth")
