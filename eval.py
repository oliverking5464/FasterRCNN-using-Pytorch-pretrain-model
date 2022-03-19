import torch
import mydataset
from PIL import Image,ImageDraw
from engine import evaluate
import utils
path = r'D:\oli\pytorch\OD\checkpoints\fasterrcnnmodel.pth'

testpath =  r'D:\oli\pytorch\OD\resistancerrronly\test\resis.txt'

model = torch.load(path)

import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    #if train:
        #transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
dataset_test = mydataset.Dataset(testpath, get_transform(train=False))
#for j in(range(len(dataset_test))):
model.eval()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)


evaluate(model, data_loader_test, device=device)