import torch
from torch import nn

from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.optim.sgd import SGD
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from SSIM import msssim, ssim
from model.SUNet import SUNet_model
from torch.utils.data import Dataset
import glob
import os
import numpy as np

class ListDataset(Dataset):
  def __init__(self, list_path, img_size=(256, 256), test=False):
    self.img_files = [list_path + img for img in glob.glob1(list_path,"*.jpg")]
    self.img_shape = img_size
    self.test = test
    self.test_len = 1000
    self.train_len = len(self.img_files) - self.test_len
    self.train_len = 10
    self.transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

  def __getitem__(self, index):

    #---------
    #  Image
    #---------
    if self.test:
      index += self.train_len
    img_path = self.img_files[index % len(self.img_files)].rstrip()
    img = Image.open(img_path).convert('RGB')
    input_img = self.transform(img)

    return input_img # , img_path

  def __len__(self):
    if self.test:
      return self.test_len
    else:
      return self.train_len


def plot_ae_outputs_den(device, net, n=10,noise_factor=0.3):
    testdataloader = torch.utils.data.DataLoader(
        ListDataset(
            # r'D:/InfoAnyagok/ArtificialIntelligence/Vision/NeuralNets/SUNet/datasets/archive/img_align_celeba/img_align_celeba/',
            r'/root/workdir/datasets/celeba/img_align_celeba/',
            img_size=(256, 256)), batch_size=1,
        shuffle=True)
    plt.figure(figsize=(16,4.5))
    for i, img in enumerate(testdataloader):

      if i == n: # for not printing entire dataset, only first couple of elements
        break

      ax = plt.subplot(3,n,i+1)
      image_noisy = add_noise(img,noise_factor)
      image_noisy = image_noisy.to(device)

      net.eval()

      with torch.no_grad():
         rec_img  = net(image_noisy)

      img = torch.permute(img, (0, 2, 3, 1))
      plt.imshow(img.cpu().squeeze().numpy())
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(3, n, i + 1 + n)
      image_noisy = torch.permute(image_noisy, (0, 2, 3, 1))
      plt.imshow(image_noisy.cpu().squeeze().numpy())
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      if i == n//2:
        ax.set_title('Corrupted images')

      ax = plt.subplot(3, n, i + 1 + n + n)
      rec_img = torch.permute(rec_img, (0, 2, 3, 1))
      plt.imshow(rec_img.cpu().squeeze().numpy())
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.7,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.3)
    plt.show()


def add_noise(inputs, noise_factor=0.3):
    noisy = inputs + torch.randn_like(inputs) * noise_factor
    noisy = torch.clip(noisy, 0., 1.)
    return noisy


if __name__ == '__main__':

    configs = {
        'IMG_SIZE': 256,
        'PATCH_SIZE': 4,
        'WIN_SIZE': 8,
        'EMB_DIM': 96,
        'DEPTH_EN': [2, 2, 2, 2],
        'HEAD_NUM': [3, 6, 12, 24],
        'MLP_RATIO': 4.0,
        'QKV_BIAS': True,
        'QK_SCALE': 8,
        'DROP_RATE': 0.,
        'ATTN_DROP_RATE': 0.,
        'DROP_PATH_RATE': 0.1,
        'APE': False,
        'PATCH_NORM': True,
        'USE_CHECKPOINTS': False,
        'FINAL_UPSAMPLE': 'Dual up-sample'
    }

    configs = {'SWINUNET': configs}

    dataloader = torch.utils.data.DataLoader(
        # ListDataset(r'D:/InfoAnyagok/ArtificialIntelligence/Vision/NeuralNets/SUNet/datasets/archive/img_align_celeba/img_align_celeba/', img_size=(256, 256)), batch_size=1, # TODO CUDA out of memory, sehogy nem akar megjavulni
        ListDataset(r'/home/adam/TokenLearner_Object_Detection/pythonProject/img_align_celeba', img_size=(256, 256)), batch_size=1, # TODO CUDA out of memory, sehogy nem akar megjavulni
        shuffle=True)
    lr = 1e-3
    net = SUNet_model(configs)
    # net = torch.load("bestmodel.pth")
    num_epoch = 20
    haveCuda = True
    if haveCuda:
        net = net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-6)
    # optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, nesterov=True, weight_decay=1e-6)  # legyen 2 nagysagrend a weight decay es lr kozott
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                    num_epoch // 2)  # mikor csokkenti -> az epoch ok felenek eltelte utan, es a felere csokkenti a lr t
    metric = 'MSSSIM'
    # criterion = nn.MSELoss()
    criterion = msssim if metric == 'MSSSIM' else ssim
    optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-05)
    noise_factor = 0.3
    bestLoss = 9999999

    scheduler = lr_scheduler.StepLR(optim, num_epoch // 2)

    train_loss = 0.0

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    traindataloader = torch.utils.data.DataLoader(
        ListDataset(
            r'D:/InfoAnyagok/ArtificialIntelligence/Vision/NeuralNets/SUNet/datasets/archive/img_align_celeba/img_align_celeba/',
            # r'/root/workdir/datasets/celeba/img_align_celeba/',
            img_size=(256, 256), test=False), batch_size=16,
        shuffle=True)

    testdataloader = torch.utils.data.DataLoader(
        ListDataset(
            r'D:/InfoAnyagok/ArtificialIntelligence/Vision/NeuralNets/SUNet/datasets/archive/img_align_celeba/img_align_celeba/',
            # r'/root/workdir/datasets/celeba/img_align_celeba/',
            img_size=(256, 256), test=True), batch_size=16,
        shuffle=True)

    for epoch in range(num_epoch):
        print('EPOCH %d/%d' % (epoch + 1, num_epoch))
        train_loss = 0.0
        for image_batch in tqdm(traindataloader):
            image_batch = image_batch.to(device)
            net.train()
            # adding noise to images
            noisy_image_batch = add_noise(image_batch, 0.3)

            denoised_image_batch = net.forward(noisy_image_batch)
            optim.zero_grad()
            loss = criterion(denoised_image_batch, image_batch)
            loss.backward()
            optim.step()
            train_loss += loss.detach().cpu().numpy()
        print('\n EPOCH {}/{} \t train loss {}'.format(epoch + 1, num_epoch, train_loss / len(traindataloader)))
        sum_val_loss = 0.0
        for image_batch in tqdm(testdataloader):
            with torch.no_grad():
                net.eval()
                image_batch = image_batch.to(device)
                noisy_image_batch = add_noise(image_batch, 0.3)

                denoised_image_batch = net.forward(noisy_image_batch)
                optim.zero_grad()
                val_loss = criterion(denoised_image_batch, image_batch)

                sum_val_loss += val_loss
        if sum_val_loss / len(testdataloader) < bestLoss:
            print("Model saved")
            bestLoss = sum_val_loss / len(testdataloader)
            torch.save(net, "bestmodel2.pth")
        print('\n EPOCH {}/{} \t val loss {}'.format(epoch + 1, num_epoch, sum_val_loss / len(testdataloader)))

        scheduler.step()
    plot_ae_outputs_den(device, net, n=10, noise_factor=0.3)
