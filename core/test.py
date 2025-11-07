
steps = [
    {'type': 'latent-guided', 'tag': 0, 'attribute': 0, 'seed': None},
    # {'type': 'latent-guided', 'tag': 1, 'attribute': 1, 'seed': None},
    # {'type': 'latent-guided', 'tag': 2, 'attribute': 1, 'seed': None},
    # {'type': 'reference-guided', 'tag': 1, 'reference': 'path-to-18.jpg'},
]

from utils import get_data_iters, prepare_sub_folder, write_loss, get_config, write_2images
from trainer import HiSD_Trainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
import numpy as np
import time

parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, default="path-to-celeba-hq.yaml")
parser.add_argument('--checkpoint', type=str,
                    default="path-to-pretrain.pt")
parser.add_argument('--input_path', type=str, default="")
parser.add_argument('--output_path', type=str, default="")
parser.add_argument('--ori_path', type=str, default="")
parser.add_argument('--output_path_smile', type=str, default="")
parser.add_argument('--output_path_gender', type=str, default="")
parser.add_argument('--output_path_young', type=str, default="")
opts = parser.parse_args()

os.makedirs(opts.output_path, exist_ok=True)
os.makedirs(opts.ori_path, exist_ok=True)

os.makedirs(opts.output_path_smile, exist_ok=True)
os.makedirs(opts.output_path_gender, exist_ok=True)
os.makedirs(opts.output_path_young, exist_ok=True)

config = get_config(opts.config)
noise_dim = config['noise_dim']
trainer = HiSD_Trainer(config)
state_dict = torch.load(opts.checkpoint)
trainer.models.gen.load_state_dict(state_dict['gen_test'])
trainer.models.gen.cuda()

E = trainer.models.gen.encode
T = trainer.models.gen.translate
G = trainer.models.gen.decode
M = trainer.models.gen.map
F = trainer.models.gen.extract

filename = time.time()
transform_list = [
    # transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(128),
    transforms.RandomCrop((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]
transform = transforms.Compose(transform_list)

if os.path.isfile(opts.input_path):
    inputs = [opts.input_path]
else:
    inputs = [os.path.join(opts.input_path, file_name) for file_name in os.listdir(opts.input_path)]
m = 0
with torch.no_grad():
    for input in inputs:
        m += 1
        x = transform(Image.open(input).convert('RGB')).unsqueeze(0).cuda()
        c = E(x)
        c_trg = c
        for j in range(len(steps)):
            step = steps[j]
            if step['type'] == 'latent-guided':
                if step['seed'] is not None:
                    torch.manual_seed(step['seed'])
                    torch.cuda.manual_seed(step['seed'])

                z = torch.randn(1, noise_dim).cuda().repeat(x.size(0), 1)
                s_trg = M(z, step['tag'], step['attribute'])
            elif step['type'] == 'reference-guided':
                reference = transform(Image.open(step['reference']).convert('RGB')).unsqueeze(0).cuda()
                s_trg = F(reference, step['tag'])

            c_trg = T(c_trg, s_trg, step['tag'])

        x_trg = G(c_trg)
        out = []
        out += [x_trg]
        write_2images(out, config['batch_size'], opts.output_path, m)

