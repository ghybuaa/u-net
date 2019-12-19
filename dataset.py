from torch.utils.data import Dataset
import PIL.Image as Image
import os


def make_dataset(root):  #  文件路径定义成列表
    imgs=[]
    n=len(os.listdir(root))//2  #返回指定文件夹下的文件数量
    for i in range(n):
        img=os.path.join(root,"%03d.png"%i)  #  原图路径
        mask=os.path.join(root,"%03d_mask.png"%i)  #  掩膜路径
        imgs.append((img,mask))
    return imgs


class LiverDataset(Dataset):  #  数据增强
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):  #  迭代列表内内容
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)
