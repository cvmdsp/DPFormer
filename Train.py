import scipy
from DPFormer import DPFormer
from CAVE_Dataset import cave_dataset
import torch.utils.data as tud
import torch
from torch.optim.lr_scheduler import MultiStepLR
import time
import datetime
import argparse
from torch.autograd import Variable
from Utils import *
import torch.nn as nn
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__=="__main__":

    class fftLoss(nn.Module):
        def __init__(self):
            super(fftLoss, self).__init__()

        def forward(self, x, y):
            diff = torch.fft.fft2(x.to(device)) - torch.fft.fft2(y.to(device))
            loss = torch.mean(abs(diff))
            return loss


    class criterion0(nn.Module):
        def __init__(self):
            super(criterion0, self).__init__()
            self.loss1 = nn.L1Loss()
            self.loss_fft = fftLoss()

        def forward(self, x, y):
            loss1 = self.loss1(x, y)
            loss_fft = self.loss_fft(x, y)
            criterion = loss1 + 0.1 * loss_fft
            return criterion

    # Model Config
    parser = argparse.ArgumentParser(description="PyTorch Code for HSI Fusion")
    parser.add_argument('--data_path', default='./Data/Train/', type=str, help='Path of the training data')
    parser.add_argument("--sizeI", default=64, type=int, help='The image size of the training patches')
    parser.add_argument("--batch_size", default=32, type=int, help='Batch size')
    parser.add_argument("--trainset_num", default=20000, type=int, help='The number of training samples of each epoch')
    parser.add_argument("--sf", default=4, type=int, help='Scaling factor')
    opt = parser.parse_args()

    def seed_torch(seed=745104):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    seed_torch()

    print(opt)

    print("===> New Model")
    model = DPFormer(4, 3, 31)

    # set the number of parallel GPUs
    print("===> Setting GPU")
    model = dataparallel(model, 1)

    # Initialize weightResult
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
        if isinstance(layer, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(layer.weight)

    # Load training data
    key = 'Train.txt'
    file_path = opt.data_path + key
    file_list = loadpath(file_path)
    HR_HSI, HR_MSI = prepare_data(opt.data_path, file_list, 20)

    # Load trained model
    initial_epoch = findLastCheckpoint(save_dir="Checkpoint/f4/DPFormer")
    if initial_epoch > 0:
        print('resuming by loading epoch %04d' % initial_epoch)
        model = torch.load(os.path.join("Checkpoint/f4/DPFormer", 'model_%04d.pth' % initial_epoch)).to(device)

    # optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0006, betas=(0.9, 0.999), eps=1e-8)
    scheduler = MultiStepLR(optimizer, milestones=list(range(1,150,5)), gamma=0.95)

    # pipline of training
    for epoch in range(initial_epoch, 200):
        model.train().to(device)

        dataset = cave_dataset(opt, HR_HSI, HR_MSI)
        loader_train = tud.DataLoader(dataset, num_workers=1, batch_size=opt.batch_size, shuffle=True)

        scheduler.step(epoch)
        epoch_loss = 0

        start_time = time.time()
        for i, (LR, RGB, HR) in enumerate(loader_train):
            LR, RGB, HR = Variable(LR), Variable(RGB), Variable(HR)
            out = model(LR.to(device), RGB.to(device))

            criterion = criterion0()
            loss = criterion(out.to(device), HR.to(device))

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('%4d %4d / %4d loss = %.10f time = %s' % (
                epoch + 1, i, len(dataset)// opt.batch_size, epoch_loss / ((i+1) * opt.batch_size), datetime.datetime.now()))

        elapsed_time = time.time() - start_time
        print('epcoh = %4d , loss = %.10f , time = %4.2f s' % (epoch + 1, epoch_loss / len(dataset), elapsed_time))

        save_dir = "Checkpoint/f4/DPFormer"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(model, os.path.join("Checkpoint/f4/DPFormer", 'model_%04d.pth' % (epoch + 1)))  # save model



