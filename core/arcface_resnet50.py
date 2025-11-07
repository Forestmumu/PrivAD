import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import torch.optim as optim
import os
import numpy as np

from pytorch_lightning.loggers import TensorBoardLogger
from arcface_models import ResNet50, ArcMarginProduct, FocalLoss
import torchvision.models as models
import pickle


pl.seed_everything(83)
import torch.nn.functional as F


def batch_misclass_rate(y_pred, y_true):
    return np.sum(y_pred != y_true) / len(y_true)


def batch_accuracy(y_pred, y_true):
    return np.sum(y_pred == y_true) / len(y_true)

# 加载预训练模型
#with open(r'C:\Users\40398\PycharmProjects\Bottleneck_Nets\lightning_logs\resnet50_scratch_weight.pkl', 'rb') as f:
#with open(r'C:\Users\Administrator\PycharmProjects\Bottleneck_Nets\lightning_logs\resnet50_scratch_weight.pkl', 'rb') as f:
with open(r'resnet50_scratch_weight.pkl', 'rb') as f:
    model_dict = pickle.load(f)
state_dict = {}
for k, v in model_dict.items():
    if isinstance(v, np.ndarray):
        v = torch.from_numpy(v)
    state_dict[k] = v
pretrained_model = models.resnet50()
pretrained_model.fc = nn.Linear(2048, 8631)
# print(model.fc.out_features)
pretrained_model.load_state_dict(state_dict)


class ArcfaceResnet50(pl.LightningModule):
    def __init__(self, in_features, out_features=10177, s=64.0, m=0.50):
        super(ArcfaceResnet50, self).__init__()

        # 加载预训练模型
        pretrained_model.fc = nn.Linear(2048, in_features)
        self.resnet50 = pretrained_model
        self.arc_margin_product = ArcMarginProduct(in_features=in_features, out_features=out_features, s=s, m=m, easy_margin=False)
        self.softmax = nn.Softmax()
        self.save_hyperparameters()
        self.criterion = FocalLoss(gamma=2)

        # roc
        self.roc = torchmetrics.ROC(task='binary')

    def forward(self, x, u):
        z = self.resnet50(x)
        output = self.arc_margin_product(z, u)
        return output, z




CHECKPOINT_PATH = os.environ.get('PATH_CHECKPOINT', 'lightning_logs/arcface_recognizer_resnet50_latent512/checkpoints/')
#os.makedirs(CHECKPOINT_PATH, exist_ok=True)

def main(model_name, Resume, save_name=None):
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """
    if save_name is None:
        save_name = model_name

    data_module = CelebaInterface(num_workers = 1,
                 dataset = 'celeba_data',
                 batch_size = 128,
                 dim_img = 224,
                 data_dir = 'D:\datasets\celeba', # 'D:\datasets\celeba'
                 sensitive_dim = 1,
                 identity_nums = 10177,
                 sensitive_attr = 'Male',
                 pin_memory=False)

    lfw_data_dir = 'E:\datasets\lfw\lfw112'
    casia_data_dir = 'E:\datasets\CASIA-FaceV5\dataset_jpg'

    lfw_casia_data = LFWCasiaInterface(dim_img=224,
                                        batch_size=128,
                                        dataset='lfw_casia_data',
                                        sensitive_attr='White',
                                        lfw_data_dir=lfw_data_dir,
                                        casia_data_dir=casia_data_dir,
                                        purpose='face_recognition')

    logger = TensorBoardLogger(save_dir=CHECKPOINT_PATH + '/lightning_logs', name='tensorboard_log')  # 把记录器放在模型的目录下面 lightning_logs\bottleneck_test_version_1\checkpoints\lightning_logs

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        callbacks=[
            ModelCheckpoint(
                mode="max",#最大
                monitor="valid_acc_epoch",#用valid_acc比较好
                dirpath=os.path.join(CHECKPOINT_PATH, 'saved_model', save_name),
                save_last=True,
                every_n_train_steps=50
            ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch"),
            EarlyStopping(
                monitor='valid_acc_epoch',#用valid_acc比较好
                patience = 5,#用valid_acc比较好5吧 间隔为3个epoch为一周期
                mode='max'#用valid_acc比较好 保存top_k 3个比较好把
            )
        ],  # Log learning rate every epoch

        default_root_dir=os.path.join(CHECKPOINT_PATH, 'saved_model', save_name),  # Where to save models
        accelerator="auto",
        devices=1,
        max_epochs=200,
        min_epochs=120,
        logger=logger,
        log_every_n_steps=10,
        precision=32,
        enable_checkpointing=True,
        check_val_every_n_epoch=3,
        fast_dev_run=False,
        reload_dataloaders_every_n_epochs=1,
        auto_lr_find=True,
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    if Resume:
        model = ArcfaceResnet50(in_features=512, out_features=10177, s=64.0, m=0.50)
        #trainer.fit(model, data_module, ckpt_path='lightning_logs/arcface_recognizer_resnet50_latent512/checkpoints/saved_model/face_recognition_resnet50/last.ckpt')
        trainer.test(model, lfw_casia_data, ckpt_path=r'E:\Bottleneck_Nets\lightning_logs\arcface_recognizer_resnet50_latent512\checkpoints\saved_model\face_recognition_resnet50\last.ckpt')
        #trainer.save_checkpoint('lightning_logs/arcface_recognizer_resnet50_latent512/checkpoints/saved_model/face_recognition_resnet50')
    else:
        resume_checkpoint_dir = os.path.join(CHECKPOINT_PATH, 'saved_models')
        os.makedirs(resume_checkpoint_dir, exist_ok=True)
        resume_checkpoint_path = os.path.join(resume_checkpoint_dir, save_name)
        print('Model will be created')
        model = ArcfaceResnet50(in_features=512, out_features=10177, s=64.0, m=0.50)
        trainer.fit(model, data_module)
        trainer.test(model, data_module)
        trainer.save_checkpoint(resume_checkpoint_path)




if __name__ == '__main__':
    main(model_name='face_recognition_resnet50',  Resume = 1, save_name=None)





