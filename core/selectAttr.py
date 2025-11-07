import os
import shutil
import math
import tqdm
import argparse

# 定义命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str,
                    default="")
parser.add_argument('--label_path', type=str,
                    default="path-to-attribute.txt")
parser.add_argument("--target_path", type=str, default="path-to-target_path")
parser.add_argument("--start", type=int, default=1002)
parser.add_argument("--end", type=int, default=3000)
opts = parser.parse_args()

target_path = opts.target_path
os.makedirs(target_path, exist_ok=True)

# 打开标签文件并读取内容
celeba_imgs = opts.img_path
celeba_label = opts.label_path

with open(celeba_label) as f:
    lines = f.readlines()

# 打开目标文件，用于写入符合条件的图片路径
with open(os.path.join(target_path, 'Selected_Attributes_female.txt'), mode='a') as output_file:
    for line in tqdm.tqdm(lines[opts.start:opts.end]):
        line = line.split()
        filename = os.path.join(os.path.abspath(celeba_imgs), line[0])

        # 检查是否满足三个属性的条件
        if int(line[21]) == -1 and int(line[40]) == 1 and int(line[32]) == 1:
            output_file.write(f'{filename}\n')
