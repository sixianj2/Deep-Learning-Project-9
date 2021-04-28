import random
import timeit

import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image

from model import CGAN_model

file_p ="data/trump2biden/train/A/0.jpg"
model_name="weights/trump2biden/netG_A2B_epoch_3.pth"
image_size=256
cuda = True

manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available():
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:0")

# create model
cgan_model = CGAN_model(100, 200, 0.0002)
generator = cgan_model.netG_A2B

# Load state dicts
generator.load_state_dict(torch.load(model_name))

# Set model mode
generator.eval()

# Load image
image = Image.open(file_p)
pre_process = transforms.Compose([transforms.Resize(image_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                  ])
image = pre_process(image).unsqueeze(0)
image = image.to(device)

start = timeit.default_timer()
fake_image = generator(image)
elapsed = (timeit.default_timer() - start)
print(f"cost {elapsed:.4f}s")
vutils.save_image(fake_image.detach(), "result.png", normalize=True)
