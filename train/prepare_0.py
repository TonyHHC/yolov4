import glob, os
import os.path
import random
import numpy as np

strOriginalPath = 'D:/Project/Tony/Python/yolov4/TX/original'
ratio = (0.6, 0.2, 0.2)

#for file in (os.listdir(os.path.join(strOriginalPath, 'B_*.bmp'))):
listB = glob.glob(os.path.join(strOriginalPath, 'B_*.bmp'))
listS = glob.glob(os.path.join(strOriginalPath, 'S_*.bmp'))
listH = glob.glob(os.path.join(strOriginalPath, 'H_*.bmp'))
print('listBuy  : {}'.format(len(listB)))
print('listSell : {}'.format(len(listS)))
print('listNone : {}'.format(len(listH)))

#listTrain = random.sample(listB, int(len(listB)*ratio[0]))
#print(listTrain)

random.shuffle(listB)
indices_for_splitting = [int(len(listB)*ratio[0]), int(len(listB)*(ratio[0]+ratio[1]))]
train, val, test = np.split(listB, indices_for_splitting)

def test(strReplace):
	strReplace = "BBB"


a = ['1','2','3']
b = ['4','5','6']
c = a + b

print(c)