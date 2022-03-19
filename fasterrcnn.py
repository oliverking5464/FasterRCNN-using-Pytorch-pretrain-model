import torchvision
import torch
import mydataset
#path of the trainset
path = r'C:\PYTHON\intern\archive\annotations\cartrain.txt'
#path of the testset
testpath =  r'C:\PYTHON\intern\archive\annotations\cartest.txt'
#path to save the model
savepath = r'C:\PYTHON\intern\pytorch\OD\checkpoints\fasterrcnnmodel_car.pth'
import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)
from engine import train_one_epoch, evaluate
import utils


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

d = mydataset.Dataset(path, get_transform(train=True))
dataset_test = mydataset.Dataset(testpath, get_transform(train=False))

indices = torch.randperm(len(d)).tolist()
d = torch.utils.data.Subset(d, indices[:-50])

data_loader = torch.utils.data.DataLoader(
    d, batch_size=1, shuffle=True, num_workers=0,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)


model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

num_epochs = 10
if __name__ == '__main__':
    
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
    
    print("That's it!")

torch.save(model,savepath)









