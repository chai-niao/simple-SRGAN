from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import Network
import Loss
from Load_Data import LoadData
import VGG
from utils import *

data_folder = './data/'
crop_size = 96
scaling_factor = 4

batch_size = 64
start_epoch = 1
epochs = 50
checkpoint = None
workers = 1
vgg19_i = 5
vgg19_j = 4
beta = 1e-3
lr = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
writer = SummaryWriter()  # command tensorboard --logdir runs


def run():
    global checkpoint, start_epoch, writer

    generator = Network.generator()
    discriminator = Network.discriminator()

    optimizerG = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr)

    VGG.TruncatedVGG19 = VGG.TruncatedVGG19(i=vgg19_i, j=vgg19_j)
    VGG.TruncatedVGG19.eval()

    content_loss_criterion = nn.MSELoss()
    adversarial_loss_criterion = nn.BCEWithLogitsLoss()

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    VGG.TruncatedVGG19 = VGG.TruncatedVGG19.to(device)
    content_loss_criterion = content_loss_criterion.to(device)

    train_dataset = LoadData(data_folder, split='train', crop_size=crop_size, scaling_factor=scaling_factor,
                             lr_img_type='imagenet-norm', hr_img_type='imagenet-norm')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                               pin_memory=True)

    for epoch in range(start_epoch, epochs + 1):
        generator.train()
        discriminator.train()

        loss_content = AverageMeter()
        loss_adversarial = AverageMeter()
        loss_discriminator = AverageMeter()

        n_iterators = len(train_loader)

        for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            sr_imgs = generator(lr_imgs)
            sr_imgs = convert_image(sr_imgs, source='[-1, 1]', target='imagenet-norm')

            sr_imgs_in_vgg_space = VGG.TruncatedVGG19(sr_imgs)
            hr_imgs_in_vgg_space = VGG.TruncatedVGG19(hr_imgs).detach()

            content_loss = content_loss_criterion(sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)

            sr_discriminated = discriminator(sr_imgs)
            adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.ones_like(sr_discriminated))

            perceptual_loss = content_loss + beta * adversarial_loss

            optimizerG.zero_grad()
            perceptual_loss.backward()

            optimizerG.step()

            loss_content.update(content_loss.item(), lr_imgs.size(0))
            loss_adversarial.update(adversarial_loss.item(), lr_imgs.size(0))

            hr_discriminated = discriminator(hr_imgs)
            sr_discriminated = discriminator(sr_imgs.detach())

            adversarial_loss = (adversarial_loss_criterion(sr_discriminated, torch.zeros_like(sr_discriminated)) +
                                adversarial_loss_criterion(hr_discriminated, torch.ones_like(hr_discriminated)))

            optimizerD.zero_grad()
            adversarial_loss.backward()

            optimizerD.step()

            loss_discriminator.update(adversarial_loss.item(), lr_imgs.size(0))

            if i == (n_iterators - 2):
                writer.add_image('epoch_' + str(epoch) + '_1', make_grid(lr_imgs[:4, :3, :, :].cpu(),
                                                                         nrow=4, normalize=True), epoch)

                writer.add_image('epoch_' + str(epoch) + '_2', make_grid(sr_imgs[:4, :3, :, :].cpu(),
                                                                         nrow=4, normalize=True), epoch)
                writer.add_image('epoch_' + str(epoch) + '_3', make_grid(hr_imgs[:4, :3, :, :].cpu(),
                                                                         nrow=4, normalize=True), epoch)

            print("第 %d / %d 个批次的损失: 内容 %.4f, 对抗 %.4f, 判别器 %.4f" % (i, n_iterators, loss_content.val,
                                                                                  loss_adversarial.val,
                                                                                  loss_discriminator.val))

        del lr.imgs, hr_imgs, sr_imgs, sr_imgs_in_vgg_space, hr_imgs_in_vgg_space, content_loss, sr_discriminated, \
            (adversarial_loss), perceptual_loss, hr_discriminated

        writer.add_scalar('Loss/Content', loss_content.val, epoch)
        writer.add_scalar('Loss/Adversarial', loss_adversarial.val, epoch)
        writer.add_scalar('Loss/Discriminator', loss_discriminator.val, epoch)

        torch.save({
            'epoch': epoch,
            'generator': generator.module.state_dict(),
            'discriminator': discriminator.module.state_dict(),
            'optimizer_g': optimizerG.state_dict(),
            'optimizer_d': optimizerD.state_dict(),
        }, 'results/checkpoint_srgan.pth')

    writer.close()


if __name__ == '__main__':
    run()
