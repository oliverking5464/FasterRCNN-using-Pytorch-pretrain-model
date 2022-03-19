import torch
import mydataset
from PIL import Image,ImageDraw
import cv2 
import numpy as np
#path of model
modelpath = '.\checkpoints\\fasterrcnnmodel_car.pth'
#path of the testset
testpath =  r'C:\PYTHON\intern\archive\annotations\cartest.txt'


model = torch.load(modelpath)
import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    #if train:
        #transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
dataset_test = mydataset.Dataset(testpath, get_transform(train=False))
color = (255,255,255)
for j in(range(len(dataset_test))):
#j =1
    img,_ = dataset_test[j]
    
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with torch.no_grad():
    
        predictionr = model([img.to(device)])
    
    imgg = np.uint8(img.mul(255).permute(1, 2, 0).byte().numpy())
    imgg = cv2.cvtColor(imgg,cv2.COLOR_RGB2BGR)

    image_h, image_w, _ = imgg.shape
    bbox_thick = int(0.6 * (image_h + image_w) / 600)
    for i in range(len(predictionr[0]['boxes'])):
        if float(predictionr[0]['scores'][i])>0.05:
            shape = ((int(predictionr[0]['boxes'][i][0]),int(predictionr[0]['boxes'][i][1])),(int(predictionr[0]['boxes'][i][2]),int(predictionr[0]['boxes'][i][3])))
            cv2.rectangle(imgg,(int(predictionr[0]['boxes'][i][0]),int(predictionr[0]['boxes'][i][1])),(int(predictionr[0]['boxes'][i][2]),int(predictionr[0]['boxes'][i][3])), color, bbox_thick)
    cv2.imshow('a',imgg)
    k = cv2.waitKey()
    if k == 27:
        cv2.destroyAllWindows()  
        break
    cv2.destroyAllWindows()     
       
    #img.save('imggr.jpg')