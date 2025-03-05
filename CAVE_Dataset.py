import torch.utils.data as tud
from Utils import *

try:
    from torch import irfft
    from torch import rfft
except ImportError:
    from torch.fft import irfft2
    from torch.fft import rfft2
    def rfft(x, d):
        t = rfft2(x, dim = (-d))
        return torch.stack((t.real, t.imag), -1)
    def irfft(x, d, signal_sizes):
        return irfft2(torch.complex(x[:,:,0], x[:,:,1]), s = signal_sizes, dim = (-d))


class cave_dataset(tud.Dataset):
    def __init__(self, opt, HR_HSI, HR_MSI, istrain = True):
        super(cave_dataset, self).__init__()
        self.path = opt.data_path
        self.istrain = istrain
        self.factor = opt.sf
        if istrain:
            self.num = opt.trainset_num
            self.file_num = 20
            self.sizeI = opt.sizeI
        else:
            self.num = opt.testset_num
            self.file_num = 12
            self.sizeI = 512
        self.HR_HSI, self.HR_MSI = HR_HSI, HR_MSI


    def __getitem__(self, index):
        if self.istrain == True:
            index1   = random.randint(0, self.file_num-1)
        else:
            index1 = index

        sigma = 2.0
        HR_HSI = self.HR_HSI[:,:,:,index1]
        HR_MSI = self.HR_MSI[:,:,:,index1]

        sz = [self.sizeI, self.sizeI]

        px      = random.randint(0, 512-self.sizeI)
        py      = random.randint(0, 512-self.sizeI)
        hr_hsi  = HR_HSI[px:px + self.sizeI:1, py:py + self.sizeI:1, :]
        hr_msi  = HR_MSI[px:px + self.sizeI:1, py:py + self.sizeI:1, :]

        if self.istrain == True:
            rotTimes = random.randint(0, 3)
            vFlip    = random.randint(0, 1)
            hFlip    = random.randint(0, 1)

            # Random rotation
            for j in range(rotTimes):
                hr_hsi  =  np.rot90(hr_hsi)
                hr_msi  =  np.rot90(hr_msi)

            # Random vertical Flip
            for j in range(vFlip):
                hr_hsi = hr_hsi[:, ::-1, :].copy()
                hr_msi = hr_msi[:, ::-1, :].copy()

            # Random horizontal Flip
            for j in range(hFlip):
                hr_hsi = hr_hsi[::-1, :, :].copy()
                hr_msi = hr_msi[::-1, :, :].copy()


        lr_hsi = cv2.GaussianBlur(hr_hsi, (5, 5), 2)  #用普通的高斯模糊
        lr_hsi = cv2.resize(lr_hsi, (sz[0] // self.factor, sz[1] // self.factor))

        hr_hsi = hr_hsi.copy().transpose(2, 0, 1)
        hr_msi = hr_msi.copy().transpose(2, 0, 1)
        lr_hsi = lr_hsi.copy().transpose(2, 0, 1)


        hr_hsi = torch.FloatTensor(hr_hsi)
        hr_msi = torch.FloatTensor(hr_msi)
        lr_hsi = torch.FloatTensor(lr_hsi)

        return lr_hsi, hr_msi, hr_hsi

    def __len__(self):
        return self.num
