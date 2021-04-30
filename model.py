import torch
import torch.nn as nn
import itertools
import random
import matplotlib.pyplot as plt


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, x):
        x = self.main(x)
        x = nn.functional.avg_pool2d(x, x.size()[2:])
        x = torch.flatten(x, 1)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Initial convolution block
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # Downsampling
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            # Residual blocks
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),

            # Upsampling
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # Output layer
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.res = nn.Sequential(nn.ReflectionPad2d(1),
                                 nn.Conv2d(in_channels, in_channels, 3),
                                 nn.InstanceNorm2d(in_channels),
                                 nn.ReLU(inplace=True),
                                 nn.ReflectionPad2d(1),
                                 nn.Conv2d(in_channels, in_channels, 3),
                                 nn.InstanceNorm2d(in_channels))

    def forward(self, x):
        return x + self.res(x)


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert (max_size > 0), "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


class CGAN_model():
    def __init__(self, decay_epochs, epochs, _lr, data_num):
        self.device = torch.device("cuda:0")
        self.netG_A2B = Generator().to(self.device)
        self.netG_B2A = Generator().to(self.device)
        self.netD_A = Discriminator().to(self.device)
        self.netD_B = Discriminator().to(self.device)

        # self.weights_init = torch.nn.init.normal_(m.weight, 0.0, 0.02)
        self.netG_A2B.apply(weights_init)
        self.netG_B2A.apply(weights_init)
        self.netD_A.apply(weights_init)
        self.netD_B.apply(weights_init)
        self.cycle_loss = torch.nn.L1Loss().to(self.device)
        self.identity_loss = torch.nn.L1Loss().to(self.device)
        self.adversarial_loss = torch.nn.MSELoss().to(self.device)
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
                                       lr=_lr, betas=(0.5, 0.999))
        self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=_lr, betas=(0.5, 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=_lr, betas=(0.5, 0.999))

        self.decay_epochs = decay_epochs
        self.epochs = epochs
        def lr_lambda(epoch):
            return 1.0 - max(0, epoch - self.decay_epochs) / (self.epochs - self.decay_epochs)
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lr_lambda)
        self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_A, lr_lambda=lr_lambda)
        self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_B, lr_lambda=lr_lambda)
        self.data_num = data_num
        self.num = 0
        self.sum_g_losses = 0
        self.sum_d_losses = 0
        self.sum_identity_losses = 0
        self.sum_gan_losses = 0
        self.sum_cycle_losses = 0

        self.g_losses = []
        self.d_losses = []
        self.identity_losses = []
        self.gan_losses = []
        self.cycle_losses = []

        self.epoch_x = [0]

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

    def train_data(self, real_image_A, real_image_B, batch_size):
        real_label = torch.full((batch_size, 1), 1, device=self.device, dtype=torch.float32)
        fake_label = torch.full((batch_size, 1), 0, device=self.device, dtype=torch.float32)
        ##############################################
        # (1) Update G network: Generators A2B and B2A
        ##############################################

        # Set G_A and G_B's gradients to zero
        self.optimizer_G.zero_grad()

        # Identity loss
        # G_B2A(A) should equal A if real A is fed
        identity_image_A = self.netG_B2A(real_image_A)
        loss_identity_A = self.identity_loss(identity_image_A, real_image_A) * 5.0
        # G_A2B(B) should equal B if real B is fed
        identity_image_B = self.netG_A2B(real_image_B)
        loss_identity_B = self.identity_loss(identity_image_B, real_image_B) * 5.0

        # GAN loss
        # GAN loss D_A(G_A(A))
        fake_image_A = self.netG_B2A(real_image_B)
        fake_output_A = self.netD_A(fake_image_A)
        loss_GAN_B2A = self.adversarial_loss(fake_output_A, real_label)
        # GAN loss D_B(G_B(B))
        fake_image_B = self.netG_A2B(real_image_A)
        fake_output_B = self.netD_B(fake_image_B)
        loss_GAN_A2B = self.adversarial_loss(fake_output_B, real_label)

        # Cycle loss
        recovered_image_A = self.netG_B2A(fake_image_B)
        loss_cycle_ABA = self.cycle_loss(recovered_image_A, real_image_A) * 10.0

        recovered_image_B = self.netG_A2B(fake_image_A)
        loss_cycle_BAB = self.cycle_loss(recovered_image_B, real_image_B) * 10.0

        # Combined loss and calculate gradients
        errG = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

        # Calculate gradients for G_A and G_B
        errG.backward()
        # Update G_A and G_B's weights
        self.optimizer_G.step()

        ##############################################
        # (2) Update D network: Discriminator A
        ##############################################

        # Set D_A gradients to zero
        self.optimizer_D_A.zero_grad()

        # Real A image loss
        real_output_A = self.netD_A(real_image_A)
        errD_real_A = self.adversarial_loss(real_output_A, real_label)

        # Fake A image loss
        fake_image_A = self.fake_A_buffer.push_and_pop(fake_image_A)
        fake_output_A = self.netD_A(fake_image_A.detach())
        errD_fake_A = self.adversarial_loss(fake_output_A, fake_label)

        # Combined loss and calculate gradients
        errD_A = (errD_real_A + errD_fake_A) / 2

        # Calculate gradients for D_A
        errD_A.backward()
        # Update D_A weights
        self.optimizer_D_A.step()

        ##############################################
        # (3) Update D network: Discriminator B
        ##############################################

        # Set D_B gradients to zero
        self.optimizer_D_B.zero_grad()

        # Real B image loss
        real_output_B = self.netD_B(real_image_B)
        errD_real_B = self.adversarial_loss(real_output_B, real_label)

        # Fake B image loss
        fake_image_B = self.fake_B_buffer.push_and_pop(fake_image_B)
        fake_output_B = self.netD_B(fake_image_B.detach())
        errD_fake_B = self.adversarial_loss(fake_output_B, fake_label)

        # Combined loss and calculate gradients
        errD_B = (errD_real_B + errD_fake_B) / 2

        # Calculate gradients for D_B
        errD_B.backward()
        # Update D_B weights
        self.optimizer_D_B.step()
        
        self.num += 1
        self.sum_g_losses += errG.item()
        self.sum_d_losses += (errD_A + errD_B).item()
        self.sum_gan_losses += (loss_GAN_A2B + loss_GAN_B2A).item()
        self.sum_cycle_losses += (loss_cycle_ABA + loss_cycle_BAB).item()
        self.sum_identity_losses += (loss_identity_A + loss_identity_B).item()

        if self.num >= self.data_num:
            self.g_losses.append(self.sum_g_losses/self.data_num)
            self.d_losses.append(self.sum_d_losses/self.data_num)
            self.identity_losses.append(self.sum_identity_losses/self.data_num)
            self.gan_losses.append(self.sum_gan_losses/self.data_num)
            self.cycle_losses.append(self.sum_cycle_losses/self.data_num)
            self.epoch_x.append(self.epoch_x[-1]+1)

            self.sum_g_losses = 0
            self.sum_d_losses = 0
            self.sum_gan_losses = 0
            self.sum_cycle_losses = 0
            self.sum_identity_losses = 0
            self.num -= self.data_num

        return errD_A, errD_B, errG, loss_identity_A, loss_identity_B, loss_GAN_A2B, loss_GAN_B2A,loss_cycle_ABA, loss_cycle_BAB


    def Update_lr(self,epoch, epochs, decay_epochs):
        # Update learning rates
        def lr_lambda(epoch):
            return 1.0 - max(0, epoch - self.decay_epochs) / (self.epochs - self.decay_epochs)
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lr_lambda)
        self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_A, lr_lambda=lr_lambda)
        self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_B, lr_lambda=lr_lambda)

    def plot_losses(self):

        fig = plt.figure()
        plt.plot(self.epoch_x[0:-1], self.g_losses)
        plt.xlabel("epoch")
        plt.ylabel("G loss")
        plt.title("G loss")
        fig.savefig('G loss.png')

        fig = plt.figure()
        plt.plot(self.epoch_x[0:-1], self.d_losses)
        plt.xlabel("epoch")
        plt.ylabel("D loss")
        plt.title("D loss")
        fig.savefig('D loss.png')

        fig = plt.figure()
        plt.plot(self.epoch_x[0:-1], self.identity_losses)
        plt.xlabel("epoch")
        plt.ylabel("Identity loss")
        plt.title("Identity loss")
        fig.savefig('Identity loss.png')

        fig = plt.figure()
        plt.plot(self.epoch_x[0:-1], self.cycle_losses)
        plt.xlabel("epoch")
        plt.ylabel("Cycle loss")
        plt.title("Cycle loss")
        fig.savefig('Cycle loss.png')

        fig = plt.figure()
        plt.plot(self.epoch_x[0:-1], self.gan_losses)
        plt.xlabel("epoch")
        plt.ylabel("GAN loss")
        plt.title("GAN loss")
        fig.savefig('GAN loss.png')
