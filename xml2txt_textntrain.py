import os 
import xml.etree.ElementTree as ET
import random
path = r'C:\PYTHON\intern\archive\annotations'

os.chdir(path)
testnames = []  
trainnames = []  
#classs = ['0 block','1 brown','2 red','3 orange','4 yellow','5 green','6 blue','7 violet','8 gray','9 white','gold','RRRR']                     
classs = ['licence']
for xml in os.listdir(path):
    if xml.endswith('.xml'):
        print(xml)
        
        tree = ET.parse(xml)
        root = tree.getroot()
        fn = root.find('filename').text
        line = path + "\\" +fn +' '
        for obj in root.findall('object'):
            name = obj.find('name').text
            bnd = obj.find('bndbox')
            xmin = bnd.find('xmin').text
            ymin = bnd.find('ymin').text
            xmax = bnd.find('xmax').text
            ymax = bnd.find('ymax').text
            
       
            for n in range(0,len(classs)):
                if name == classs[n]:
                    clas = n
                    if clas != 11 :
                        na = str(clas)
                        na = '11'
                        line = line + xmin+','+ymin +','+xmax+','+ymax+','+na+' '
        ra = random.randint(1, 3)
        if ra >1:
            trainnames.append(line)
        else:
            testnames.append(line)
    else:
        pass

with open('cartrain.txt', 'w') as f:
    for item in trainnames:
        f.write("%s\n" % item)
with open('cartest.txt', 'w') as f:
    for item in testnames:
        f.write("%s\n" % item)
