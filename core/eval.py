
import os
from deepface import DeepFace
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.image as mpimg
from tqdm import tqdm
from PIL import Image

CelebA_id_path = r"path-to-identity_CelebA.txt"
CelebAHQ_mapping_path = r"path-to-CelebA-HQ-to-CelebA-mapping.txt"
CelebA_img_dir = r"path-to-celeba_image"
CelebAHQ_img_dir = r"path-to-celeba_HQ_image"

def get_CelebA_ID_mapping_images():
    CelebA_ID_mapping_images = {}
    with open(CelebA_id_path, "r") as f:
        for line in f.readlines():
            line = line.split()
            if line[1] not in CelebA_ID_mapping_images:
                CelebA_ID_mapping_images[line[1]] = [line[0]]
            else:
                CelebA_ID_mapping_images[line[1]].append(line[0])
    return CelebA_ID_mapping_images

def get_CelebA_image_mapping_ID():
    CelebA_image_mapping_ID = {}
    with open(CelebA_id_path, "r") as f:
        for line in f.readlines():
            line = line.split()
            CelebA_image_mapping_ID[line[0]] = line[1]
    return CelebA_image_mapping_ID

def get_CelebAHQ_image_mapping_ID():
    celeba_image_id = get_CelebA_image_mapping_ID()
    CelebAHQ_image_mapping_ID = {}
    with open(CelebAHQ_mapping_path, "r") as f:
        f.readline()
        for line in f.readlines():
            line = line.split()
            image = line[0] + '.jpg'
            id = celeba_image_id[line[2]]
            if image not in CelebAHQ_image_mapping_ID:
                CelebAHQ_image_mapping_ID[image] = id
    return CelebAHQ_image_mapping_ID

def get_CelebAHQ_ID_mapping_images():
    celeba_image_mapping_ID = get_CelebA_image_mapping_ID()
    CelebAHQ_ID_mapping_images = {}
    with open(CelebAHQ_mapping_path, "r") as f:
        f.readline()
        for line in f.readlines():
            line = line.split()
            image = line[0] + '.jpg'
            celeba = line[2]
            id = celeba_image_mapping_ID[celeba]
            if id not in CelebAHQ_ID_mapping_images:
                CelebAHQ_ID_mapping_images[id] = [image]
            else:
                CelebAHQ_ID_mapping_images[id].append(image)
    return CelebAHQ_ID_mapping_images

# 使用 DeepFace 计算图像对之间的相似度
def compute_similarity(image_path1, image_path2, model="VGG-Face"):
    # 使用 DeepFace 库比较两张人脸的相似性
    result = DeepFace.verify(img1_path=image_path1, img2_path=image_path2, model_name=model, enforce_detection=False)
    return result["distance"], result["verified"]

def eval_id_utility(experiment, image_paths, mode='CelebA', test_epoch=10, image_num=10, model='VGG-Face'):
    if mode == 'CelebA':
        image_mapping_ID = get_CelebA_image_mapping_ID()
        ID_mapping_images = get_CelebA_ID_mapping_images()
        img_dir = CelebA_img_dir
    else:
        ID_mapping_images = get_CelebAHQ_ID_mapping_images()
        image_mapping_ID = get_CelebAHQ_image_mapping_ID()
        img_dir = CelebAHQ_img_dir

    imgs = os.listdir(image_paths)
    epoch = 0
    while epoch < test_epoch:
        score, true = [], []
        for i,img in tqdm(enumerate(imgs), total=len(imgs)):
            protect_img = os.path.join(image_paths, img)
            id = image_mapping_ID[img]
            n = image_num if image_num >= len(ID_mapping_images[id]) else len(ID_mapping_images[id])
            for image in ID_mapping_images[id][:n]:
                ori = os.path.join(img_dir, image)
                distance, _ = compute_similarity(protect_img, ori, model)
                score.append(1 - distance)
                true.append(True)

            for j in range(n):
                while True:
                    if mode == 'CelebA':
                        r = str(random.randint(1, 202599)).zfill(6) + '.jpg'
                    else:
                        r = str(random.randint(0, 29999)) + '.jpg'
                    if image_mapping_ID[r] != id:
                        true.append(False)
                        break
                r = os.path.join(img_dir, r)
                distance, _ = compute_similarity(protect_img, r, model)
                score.append(1 - distance)
        score = np.array(score)
        true = np.array(true)
        fpr, tpr, ADNet_thresholds = roc_curve(true,score)
        id_auc = auc(fpr, tpr)
        name = image_paths.split('//')[-1]
        print(f"[{epoch+1}/{test_epoch}] {id_model}-{experiment}-{name} AUV: ", id_auc)
        save_dir = f"result//{experiment}"
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, f"{id_model}-{name}.npz")
        np.savez_compressed(out, score=np.asarray(score), true=np.asarray(true))

        fpr, tpr, thresholds = roc_curve(true, score)
        auc_id = auc(fpr, tpr)
        frr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute(frr - fpr))]
        eer = fpr[np.nanargmin(np.absolute(frr - fpr))]
        print(f"<{id_model}-{experiment}-{name}>eer_threshold:{eer_threshold},eer:{eer}")

        # 设定阈值
        threshold = eer_threshold  # 可用 ROC/EER 找最优阈值
        # 预测结果：分数 > 阈值 → 同一个人
        predicted_labels = [1 if s > threshold else 0 for s in score]
        # 构造混淆矩阵
        tn, fp, fn, tp = confusion_matrix(true, predicted_labels).ravel()
        # 计算 FMR 和 FNMR
        fmr = fp / (fp + tn)  # 错误接受率（受骗）
        fnmr = fn / (fn + tp)  # 错误拒绝率（拒真）
        print(f"<{id_model}-{experiment}-{name}> FMR: {fmr:.4f}, FNMR: {fnmr:.4f}")
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        print(f"<{id_model}-{experiment}-{name}> accuracy: {accuracy:.4f}")
        out = os.path.join(save_dir, f"{id_model}-{name}.txt")
        with open(out, "w") as f:
            f.write(f"<{id_model}-{experiment}-{name}> AUC: {id_auc}")
            f.write(f"<{id_model}-{experiment}-{name}> eer_threshold:{eer_threshold},eer:{eer}")
            f.write(f"<{id_model}-{experiment}-{name}> FMR: {fmr:.4f}, FNMR: {fnmr:.4f}")
            f.write(f"<{id_model}-{experiment}-{name}> accuracy: {accuracy:.4f}")

        epoch += 1

if __name__ == '__main__':

    id_models = ['VGG-Face' , 'ArcFace', 'Facenet', 'OpenFace']
    #attr = [['female', 'male'], ['old', 'young'], ['smile_with', 'without']]
    exps = ['VecGAN','Rapp' ] # 实验项目
    for exp in exps:
        for id_model in id_models:
            attrs = [['female', 'male'], ['old', 'young'],['smile','wosmile']]
            for attr in attrs:
                for i,_ in enumerate(attr):
                    imgs_dir = f'result//{exp}//{attr[i%2]}2{attr[(i+1)%2]}'
                    eval_id_utility(experiment=exp, image_paths=imgs_dir,
                                    mode='CelebAHQ', test_epoch=1, image_num=10,
                                    model=id_model)

