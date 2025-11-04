import os
import shutil
import math
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default="path-to-img")
parser.add_argument('--label_path', type=str,default="path-to-attribute.txt")
parser.add_argument("--target_path", type=str, default="path-to-attr_trg")
parser.add_argument("--start", type=int, default=3002)
parser.add_argument("--end", type=int, default=30002)
opts = parser.parse_args()

target_path = opts.target_path

os.makedirs(target_path, exist_ok=True)

Tags_Attributes = {
    'Male': ['with', 'without'],
    'Young': ['with', 'without'],
    'Smiling': ['with', 'without'],
}

for tag in Tags_Attributes.keys():
    for attribute in Tags_Attributes[tag]:
        open(os.path.join(target_path, f'{tag}_{attribute}.txt'), 'w')

# celeba-hq
celeba_imgs = opts.img_path
celeba_label = opts.label_path

with open(celeba_label) as f:
    lines = f.readlines()

for line in tqdm.tqdm(lines[opts.start:opts.end]):

    line = line.split()

    filename = os.path.join(os.path.abspath(celeba_imgs), line[0])

    # Use only gender and age as tag-irrelevant conditions. Add other labels if you want.
    if int(line[21]) == 1:
        with open(os.path.join(target_path, 'Male_with.txt'), mode='a') as f:
            f.write(f'{filename} {line[6]} {line[16]}\n')
    elif int(line[21]) == -1:
        with open(os.path.join(target_path, 'Male_without.txt'), mode='a') as f:
            f.write(f'{filename} {line[6]} {line[16]}\n')

    if  int(line[40]) == 1:
        with open(os.path.join(target_path, 'Young_with.txt'), mode='a') as f:
            f.write(f'{filename} {line[6]} {line[16]}\n')
    elif int(line[40]) == -1:
        with open(os.path.join(target_path, 'Young_without.txt'), mode='a') as f:
            f.write(f'{filename} {line[6]} {line[16]}\n')

    if  int(line[32]) == 1:
        with open(os.path.join(target_path, 'Smiling_with.txt'), mode='a') as f:
            f.write(f'{filename} {line[6]} {line[16]}\n')
    elif int(line[32]) == -1:
        with open(os.path.join(target_path, 'Smiling_without.txt'), mode='a') as f:
            f.write(f'{filename} {line[6]} {line[16]}\n') #  有无刘海，有没有戴眼镜
            


    


