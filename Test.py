import torch.utils.data as tud
import argparse
from Utils import *
from CAVE_Dataset import cave_dataset
from thop import profile
import scipy.io

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="PyTorch Code for HSI Fusion")
parser.add_argument('--data_path', default='./Data/Test/', type=str, help='path of the testing data')
parser.add_argument("--sizeI", default=512, type=int, help='the size of trainset')
parser.add_argument("--testset_num", default=12, type=int, help='total number of testset')
parser.add_argument("--batch_size", default=1, type=int, help='Batch size')
parser.add_argument("--sf", default=4, type=int, help='Scaling factor')
parser.add_argument("--kernel_type", default='gaussian_blur', type=str, help='Kernel type')
opt = parser.parse_args()
print(opt)

key = 'Test.txt'
file_path = opt.data_path + key
file_list = loadpath(file_path, shuffle=False)
HR_HSI, HR_MSI = prepare_data(opt.data_path, file_list, 12)

dataset = cave_dataset(opt, HR_HSI, HR_MSI, istrain=False)
loader_train = tud.DataLoader(dataset, batch_size=opt.batch_size)

model = torch.load("./Checkpoint/f4/DPFormer/model_0196.pth")
model = model.eval()
model = model.to(device)


result_dir = './Result/f4/DPFormer/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

for j, (LR, RGB, HR) in enumerate(loader_train):
    with torch.no_grad():
        out = model(LR.to(device), RGB.to(device))
        result = out
        result = result.clamp(min=0., max=1.)
        result_np = result.cpu().detach().numpy()
        result_combined = np.concatenate(result_np, axis=1)
        scipy.io.savemat(f'./Result/f4/DPFormer/'+file_list[j]+'.mat', {'result': result_combined})

flops, params = profile(model, inputs=(LR.to(device), RGB.to(device),))

print('params(M),', params / 1e6)
print('flops(G),', flops / 1e9)
