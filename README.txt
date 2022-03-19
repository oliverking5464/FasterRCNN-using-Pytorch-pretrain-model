just learning from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
if there is anythings wrong, please tell me ^^

to make dataset:
use xml2txt_textntrain.py to make dataset 

dataloader : mydataset.py don't need to change

train :

change fastrcnn.py or ssd.py or retinanet.py  line6,7 to set dataset path(txt file made by xml2txt)

	path = r'D:\oli\pytorch\OD\resistancerrronly\train\resis.txt'
	testpath =  r'D:\oli\pytorch\OD\resistancerrronly\test\resis.txt'


change fastrcnn.py or ssd.py or retinanet.py  line9 to set save path
	
	savepath = r'D:\oli\pytorch\OD\checkpoints\fasterrcnnmodel_3.pth'




detect :

change detect.py line6 to set model path
	path = r'D:\oli\pytorch\OD\checkpoints\fasterrcnnmodel.pth'


change detect.py line8 to set test dataset path
	testpath =  r'D:\oli\pytorch\OD\resistancerrronly\test\resis.txt'

evaluate :

change eval.py line6 to set model path
	path = r'D:\oli\pytorch\OD\checkpoints\fasterrcnnmodel.pth'

change eval.py line 8 to set model dataset path
	testpath =  r'D:\oli\pytorch\OD\resistancerrronly\test\resis.txt'

change eval.py line 29 to select model to test
	evaluate(model_you_selected, data_loader_test, device=device)
