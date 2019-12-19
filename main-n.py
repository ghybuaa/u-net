import torch
import argparse  #  argparse，模块用于解析命令行参数，如  python  main.py  test  --port
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from unet import Unet
from dataset import LiverDataset
from tensorboardX import SummaryWriter
import numpy as np

import multiprocessing
multiprocessing.set_start_method('spawn', True)


# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  #  标准化至[-1,1]，规定均值和标准差
])

iou_thresholds = torch.tensor([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

def iou(img_true, img_pred):
     img_pred = (img_pred > 0).float()
     i = (img_true * img_pred).sum()
     u = (img_true + img_pred).sum()
     return i / u if u != 0 else u

def iou_metric(imgs_pred, imgs_true):
    num_images = len(imgs_true)
    scores = np.zeros(num_images)
    for i in range(num_images):
        if imgs_true[i].sum() == imgs_pred[i].sum() == 0:
            scores[i] = 1
        else:
            scores[i] = (iou_thresholds+iou(imgs_true[i], imgs_pred[i])).mean()
    return scores.mean()

# mask只需要转换为tensor
y_transforms = transforms.ToTensor()

def train_model(model, criterion, optimizer, dataload, num_epochs=20):
    writer = SummaryWriter('runs')
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # 梯度归零
            optimizer.zero_grad()
            # forward，前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)  #  计算损失
            writer.add_scalar('train', loss, epoch)
            loss.backward()  #  梯度下降，计算梯度
            optimizer.step()  #  更新参数
            epoch_loss += loss.item()
            miou = iou_metric(inputs, labels)
            print("%d/%d,train_loss:%0.3f,miou:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item(),miou))
        print("epoch %d loss:%0.3f " % (epoch, epoch_loss/step,))
        writer.close()
    torch.save(model.state_dict(), 'weights_%d.pth' % epoch)  #  保存参数模型
    return model


#训练模型
def train(args):
    model = Unet(3, 1).to(device)  #  输入3通道，输出1通道
    batch_size = args.batch_size
    criterion = nn.BCEWithLogitsLoss()   #  损失函数
    optimizer = optim.Adam(model.parameters())  #  获得模型的参数
    liver_dataset = LiverDataset("data/train",transform=x_transforms,target_transform=y_transforms) 
    #  加载数据集，返回的是一对原图+掩膜，和所有图片的数目
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    #  DataLoader接口是自定义数据接口输出输入接口，将已有的数据输入按照batch size封装成Tensor
    #  batch_size=4,epoch=10,共100个minbatch
    # shuffle，每个epoch将数据打乱
    # num_workers: 多个进程倒入数据，加速倒入速度
    train_model(model, criterion, optimizer, dataloaders)  # 训练

#显示模型的输出结果
def test(args):
    model = Unet(3, 1)
    model.load_state_dict(torch.load(args.ckpt,map_location='cpu'))
    liver_dataset = LiverDataset("data/val", transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()
    import matplotlib.pyplot as plt
    plt.ion()
    with torch.no_grad():
        for x, _ in dataloaders:
            y=model(x)
            img_y=torch.squeeze(y).numpy()
            plt.imshow(img_y)
            plt.pause(0.01)
        plt.show()


if __name__ == '__main__':
    #参数解析
    parse = argparse.ArgumentParser()  # 创建一个ArgumentParser对象
    parse.add_argument("action", type=str, help="train or test")  #  添加参数
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument("--ckpt", type=str, help="the path of model weight file")
    args = parse.parse_args()

    if args.action=="train":    
        train(args)
    elif args.action=="test":  
        test(args)
