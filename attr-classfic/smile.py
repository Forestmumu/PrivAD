import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.models import ResNet18_Weights
from tqdm import tqdm
class CelebHQDataset(Dataset):
    def __init__(self, textfile, root_dir, transform=True):

        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.labels = []

        with open(textfile, 'r') as file:
            lines = file.readlines()
            for line in lines[2:]:
                parts = line.strip().split()
                filename = parts[0]  # 文件名
                bang_label = int(parts[32])  # 微笑标签

                # 将 -1 转换为 0
                if bang_label == -1:
                    bang_label = 0

                self.image_files.append(filename)
                self.labels.append(bang_label)

        self.image_files = sorted(os.listdir(self.root_dir), key=lambda x: int(x.split('.')[0]))
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]  # 获取图像文件名
        img_path = os.path.join(self.root_dir, image_name)  # 构建图像路径
        image = Image.open(img_path)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)  # 根据文件名查找标签

        if self.transform:
            transform_list = [
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(128),
                transforms.RandomCrop((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
            transform = transforms.Compose(transform_list)
            image = transform(image)


        return image, label





def main():
    root_dir = 'path-to-imgge'
    attr_txt = 'path-to-attribute.txt'

    full_dataset = CelebHQDataset(textfile=attr_txt, root_dir=root_dir, transform=True)

    # 手动分割为训练集和测试集
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [27000, 3000])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=1,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=1)

    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)  # 输出1个神经元，表示性别

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()  # 二分类任务使用 BCEWithLogitsLoss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 早停参数
    early_stopping_patience = 5  # 如果10个epoch准确率没有提升则停止训练
    best_accuracy = 0.0
    epochs_no_improve = 0
    print('Training is bengin')
    num_epochs = 500  # 设置一个较大的上限
    for epoch in range(num_epochs):
        print(f'This is epoch {epoch}')
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader,desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 验证集测试
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze(1)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')
        accuracy_str = f"{accuracy:.4f}"  # 将 accuracy 保留四位小数
        model_path = os.path.join('D:\\wangy\\HiSD-self\\attr-classfic\\smile-model',f'model_epoch_{epoch + 1}_acc_{accuracy_str}.pth')
        torch.save(model.state_dict(), model_path)

        # 检查是否需要早停
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break


if __name__ == '__main__':
    main()
