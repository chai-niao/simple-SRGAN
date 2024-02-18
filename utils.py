import json
import os
import random
import torch
import torchvision.transforms.functional as FT
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)
imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
imagenet_mean_cuda = torch.FloatTensor([0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
imagenet_std_cuda = torch.FloatTensor([0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)


def create_data_lists(train_folders, test_folders, min_size, output_folder):

    train_images = list()
    for d in train_images:
        for i in os.listdir(d):
            img_path = os.path.join(d, i)
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                train_images.append(img_path)

    with open(os.path.join(output_folder, 'train_images.json'), 'w') as j:
        json.dump(train_images, j)

    for d in test_folders:
        test_images = list()
        test_name = d.split("/")[-1]
        for i in os.listdir(d):
            img_path = os.path.join(d, i)
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                test_images.append(img_path)

        with open(os.path.join(output_folder, test_name + '_test_images.json'), 'w') as j:
            json.dump(test_images, j)

    print("Generation complete. The training and testing set file lists have been saved in %s" % output_folder)

def convert_image(img, source, target):
    assert source in {'pil', '[0, 1]', '[-1, 1]'}, "source should be 'pil', '[0, 1]', or '[-1, 1]'"
    assert target in {'pil', '[0, 1]', '[0, 255]', '[-1, 1]', 'imagenet-norm', 'y-channel'}, "target should be 'pil', '[0, 1]', '[0, 255]', '[-1, 1]', 'imagenet-norm', or 'y-channel'"

    if source == 'pil':
        img = FT.to_tensor(img)
    elif source == '[0, 1]':
        pass  # already in correct format
    elif source == '[-1, 1]':
        img = (img + 1.) / 2.

    if target == 'pil':
        img = FT.to_pil_image(img)
    elif target == '[0, 1]':
        pass
    elif target == '[0, 255]':
        img = img * 255.
    elif target == '[-1, 1]':
        img = img * 2. - 1.
    elif target == 'imagenet-norm':
        if img.ndimension() == 3:
            img = (img - imagenet_mean) / imagenet_std
        elif img.ndimension() == 4:
            img = (img - imagenet_mean_cuda) / imagenet_std_cuda
    elif target == 'y-channel':
        img = torch.matmul(255. * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :],
                           rgb_weights) / 255. + 16.
    return img

class ImageTransforms(object):

    def __init__(self, split, crop_size, scaling_factor, lr_img_type,
                 hr_img_type):
        self.split = split.lower()
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type

        assert self.split in {'train', 'test'}

    def __call__(self, img):
        # 裁剪
        if self.split == 'train':
            # 从原图中随机裁剪一个子块作为高分辨率图像
            left = random.randint(1, img.width - self.crop_size)
            top = random.randint(1, img.height - self.crop_size)
            right = left + self.crop_size
            bottom = top + self.crop_size
            hr_img = img.crop((left, top, right, bottom))
        else:
            # 从图像中尽可能大的裁剪出能被放大比例整除的图像
            x_remainder = img.width % self.scaling_factor
            y_remainder = img.height % self.scaling_factor
            left = x_remainder // 2
            top = y_remainder // 2
            right = left + (img.width - x_remainder)
            bottom = top + (img.height - y_remainder)
            hr_img = img.crop((left, top, right, bottom))

        # 下采样
        lr_img = hr_img.resize((int(hr_img.width / self.scaling_factor),
                                int(hr_img.height / self.scaling_factor)),
                               Image.BICUBIC)

        # 安全性检查
        assert hr_img.width == lr_img.width * self.scaling_factor and hr_img.height == lr_img.height * self.scaling_factor

        # 转换图像
        lr_img = convert_image(lr_img, source='pil', target=self.lr_img_type)
        hr_img = convert_image(hr_img, source='pil', target=self.hr_img_type)

        return lr_img, hr_img


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(state, filename):
    torch.save(state, filename)