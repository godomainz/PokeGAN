import torch
from aegan import Generator as G
import torchvision.utils as vutils
import os

device = torch.device('cpu')
netG = G()
netG.load_state_dict(torch.load('trained_generator_weights.pt', map_location=device))
num_of_imgs = 50
vec = torch.randn((num_of_imgs, 16))
with torch.no_grad():
    fake = netG(vec)

directory='fakes'
if not os.path.exists(directory):
    os.makedirs(directory)
starting_num = 1
for i in range(num_of_imgs):
    vutils.save_image(fake.data[i], os.path.join(directory, f'aipoke{starting_num:02d}.png'), normalize=True)
    starting_num = starting_num + 1