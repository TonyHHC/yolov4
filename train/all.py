import glob, os
import os.path
import shutil
import subprocess
import time
import re
from shutil import copyfile
import shutil
import cv2
from xml.dom import minidom
from os.path import basename
from tqdm import tqdm
import random
import numpy as np

#--------------------------------------------------------------------
strParameterFile = 'D:/Project/Tony/Python/yolov4/TX/parameter.xml'
strLabelingXMLTemplate = 'D:/Project/Tony/Python/yolov4/TX/labeling_template.xml'
folder_classes = ('train', 'validate', 'test')
cfgs_total = {
	"yolov4": ["cfg/yolov4.cfg", "pretrained/yolov4.conv.137"],
	"yolov4-tiny": ["cfg/yolov4-tiny.cfg", "pretrained/yolov4-tiny.conv.29"],
}
ratio = (0.6, 0.2, 0.2)

originalFolder = ''
xmllabelFolder = {}
imgFolder = {}
txtlabelFolder = {}
txtFile = {}
classList = {}
darkhome = ''
namesFile = ''
datasFile = ''
traincommandfile = ''
configFile = ''
weightsFolder = ''
config = ''
pretrained = ''
batch = ''
subdivisions = ''
width = ''
height = ''
numclusters = []
anchors = ''
#--------------------------------------------------------------------

def getXmlChildNodeValue(nodeParent, tag):
	tmpNode = nodeParent.getElementsByTagName(tag)[0]
	try:
		return tmpNode.childNodes[0].nodeValue
	except:
		return None

def ReadParameter(strFilename='00_parameter.xml'):
	print("--> processing {}".format(strFilename))
	
	root = minidom.parse(strFilename)
	iClassCount = 0
	
	for eachClass in folder_classes:
		tvNode = root.getElementsByTagName(eachClass)[0]
	
		xmllabelFolder[eachClass] = getXmlChildNodeValue(tvNode, 'xmllabelFolder')
		imgFolder[eachClass] = getXmlChildNodeValue(tvNode, 'imgFolder')
		txtlabelFolder[eachClass] = getXmlChildNodeValue(tvNode, 'txtlabelFolder')
		txtFile[eachClass] = getXmlChildNodeValue(tvNode, 'txtFile')
	
	classNode = root.getElementsByTagName("class")[0]
	nameNodes = classNode.getElementsByTagName("name")
	for nameNode in nameNodes:
		classList[nameNode.childNodes[0].nodeValue] = iClassCount
		iClassCount += 1
		
	tmpNode = root.getElementsByTagName("namesFile")[0]
	namesFile = tmpNode.childNodes[0].nodeValue
	
	tmpNode = root.getElementsByTagName("datasFile")[0]
	datasFile = tmpNode.childNodes[0].nodeValue
	
	tmpNode = root.getElementsByTagName("traincommandfile")[0]
	traincommandfile = tmpNode.childNodes[0].nodeValue
	
	tmpNode = root.getElementsByTagName("configFile")[0]
	configFile = tmpNode.childNodes[0].nodeValue
	
	tmpNode = root.getElementsByTagName("weightsFolder")[0]
	weightsFolder = tmpNode.childNodes[0].nodeValue
	
	tmpNode = root.getElementsByTagName("originalFolder")[0]
	originalFolder = tmpNode.childNodes[0].nodeValue
	
	darkhome = getXmlChildNodeValue(root, 'darkhome')
	config = getXmlChildNodeValue(root, 'config')
	pretrained = getXmlChildNodeValue(root, 'pretrained')
	batch = getXmlChildNodeValue(root, 'batch')
	subdivisions = getXmlChildNodeValue(root, 'subdivisions')
	width = getXmlChildNodeValue(root, 'width')
	height = getXmlChildNodeValue(root, 'height')
	numclusters = getXmlChildNodeValue(root, 'numclusters')
	
	return namesFile, datasFile, configFile, traincommandfile, originalFolder, weightsFolder, xmllabelFolder, imgFolder, txtlabelFolder, txtFile, darkhome, config, pretrained, batch, subdivisions, width, height, numclusters, classList
	
def transferYolo( xmlFilepath, img_h, img_w ):

	print("--> processing {}".format(xmlFilepath))
	
	if(os.path.isfile(xmlFilepath)):

		try:
			#print("minidom.parse " + xmlFilepath)
			labelXML = minidom.parse(xmlFilepath)
		except:
			return

		labelName = []
		labelXmin = []
		labelYmin = []
		labelXmax = []
		labelYmax = []
		totalW = 0
		totalH = 0
		countLabels = 0

		tmpArrays = labelXML.getElementsByTagName("filename")
		for elem in tmpArrays:
			filenameImage = elem.firstChild.data

		tmpArrays = labelXML.getElementsByTagName("name")
		for elem in tmpArrays:
			labelName.append(str(elem.firstChild.data))

		tmpArrays = labelXML.getElementsByTagName("xmin")
		for elem in tmpArrays:
			labelXmin.append(int(elem.firstChild.data))

		tmpArrays = labelXML.getElementsByTagName("ymin")
		for elem in tmpArrays:
			labelYmin.append(int(elem.firstChild.data))

		tmpArrays = labelXML.getElementsByTagName("xmax")
		for elem in tmpArrays:
			labelXmax.append(int(elem.firstChild.data))

		tmpArrays = labelXML.getElementsByTagName("ymax")
		for elem in tmpArrays:
			labelYmax.append(int(elem.firstChild.data))
			
		i = 0
		x = (labelXmin[i] + (labelXmax[i]-labelXmin[i])/2) * 1.0 / img_w
		y = (labelYmin[i] + (labelYmax[i]-labelYmin[i])/2) * 1.0 / img_h
		w = (labelXmax[i]-labelXmin[i]) * 1.0 / img_w
		h = (labelYmax[i]-labelYmin[i]) * 1.0 / img_h
		
		print('--> (x, y, w, h) : ({}, {}, {}, {})'.format(x, y, w, h))
		
		return x, y, w, h
		
def copyimgandyololabeltext(eachImgFile, imgFolder, x, y, w, h):
	srcImg = eachImgFile
	filepath, file_extension = os.path.splitext(srcImg)
	imgFilebasename = os.path.basename(srcImg)
	imgFilebasenamewithnoext = imgFilebasename.split('.')[0]
	tgtImg = os.path.join(imgFolder, imgFilebasename)
	tgtYoloLabelTxt = os.path.join(imgFolder, imgFilebasenamewithnoext) + '.txt'
	print('--> tgtImg:{}, tgtYoloLabelTxt:{}'.format(tgtImg, tgtYoloLabelTxt))
	
	# copy
	shutil.copy(srcImg, tgtImg)
	
	#
	classID = 0
	if imgFilebasename[0] == 'H':
		classID = 0
	if imgFilebasename[0] == 'B':
		classID = 1
	if imgFilebasename[0] == 'S':
		classID = 2
	with open(tgtYoloLabelTxt, 'w') as the_file:
		the_file.write(str(classID) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n')
	the_file.close
		
def create_folder():
	for key in folder_classes:
		objsTmp = [imgFolder[key],txtlabelFolder[key], xmllabelFolder[key]]
		for eachObj in objsTmp:
			folder = eachObj
			if folder is not None:
				print("--> create {}".format(folder))
				if not os.path.exists(folder):
					os.makedirs(folder)
				files = glob.glob(os.path.join(folder,'*.*'))
				for file in files:
					os.remove(file)
				
	tmpFolder = os.path.dirname(configFile)
	print("--> create {}".format(tmpFolder))
	if not os.path.exists(tmpFolder):
		os.makedirs(tmpFolder)

	print("--> create {}".format(weightsFolder))
	if not os.path.exists(weightsFolder):
		os.makedirs(weightsFolder)

def create_names(namesFile, classList):
	with open(namesFile, 'w') as the_file:
		for key in classList:
			print('--> add {}'.format(key))
			the_file.write(key + '\n')
	the_file.close()
	
def create_txt(imgFolder, txtfilePath):
	print('--> processing {}'.format(txtfilePath))
	with open(txtfilePath, 'w') as the_file:
		for file in (os.listdir(imgFolder)):
			filename, file_extension = os.path.splitext(file)
			file_extension = file_extension.lower()
			if(file_extension == ".jpg" or file_extension==".png" or file_extension==".jpeg" or file_extension==".bmp"):
				imgfile = os.path.join(imgFolder, file)
				the_file.write(imgfile + '\n')
				
	the_file.close()
	
def create_data(datasFilePath, classList, txtFile, namesFile, weightsFolder):
	print('--> processing {}'.format(datasFilePath))
	with open(datasFilePath, 'w') as the_file:
		the_file.write("classes= " + str(len(classList)) + "\n")
		the_file.write("train  = " + txtFile['train'] + "\n")
		the_file.write("valid  = " + txtFile['validate'] + "\n")
		the_file.write("names  = " + namesFile + "\n")
		the_file.write("backup = " + weightsFolder + "/")
	the_file.close()
	
def create_config(darkhome, config, configFile, datasFile, classList, batch, subdivisions, width, height, anchors):
	print('--> processing {}'.format(configFile))
	
	srcConfigPath = os.path.join(darkhome, cfgs_total[config][0])
	with open(srcConfigPath) as file:
		file_content = file.read()
	file.close
	
	filters = (len(classList)+5) * 3
	
	file_content = re.sub(r'classes\s*=.*', 'classes='+str(len(classList)), file_content)
	file_content = re.sub(r'filters\s*=.*', 'filters='+str(filters), file_content)
	file_content = re.sub(r'batch\s*=.*', 'batch='+str(batch), file_content)
	file_content = re.sub(r'subdivisions\s*=.*', 'subdivisions='+str(subdivisions), file_content)
	file_content = re.sub(r'width\s*=.*', 'width='+str(width), file_content)
	file_content = re.sub(r'height\s*=.*', 'height='+str(height), file_content)
	max_batches = len(classList)*2000
	file_content = re.sub(r'max_batches\s*=.*', 'max_batches='+str(max_batches), file_content)
	file_content = re.sub(r'steps\s*=.*', 'steps='+str(int(max_batches*0.8))+','+str(int(max_batches*0.9)), file_content)
	file_content = re.sub(r'anchors\s*=.*', anchors, file_content)
	
	with open(configFile, 'w') as the_file:
		the_file.write(file_content)
	the_file.close
	
def cauculate_anchors(darkhome, datasFile, numclusters, width, height):
	print('--> processing {}, {}, {}, {}'.format(datasFile, numclusters, width, height))
	
	darknet_path = os.path.join(darkhome, 'darknet')
	nmap_out = subprocess.run([darknet_path, 'detector', 'calc_anchors', datasFile, '-num_of_clusters', \
		str(numclusters), '-width', str(width), '-height', str(height)], universal_newlines=False, stdout=subprocess.PIPE)
		
	nmap_lines = nmap_out.stdout.splitlines()
	
	anchors = 'not found.'
	for line in nmap_lines:
		if 'anchors =' in str(line):
			anchors = line.decode('ascii')
			
	print('--> ' + anchors)
	
	return anchors
	
	
def show_train_command(darkhome, configFile, datasFile, pretrained, traincommandfile):
	print('--> preparing command and save to {}'.format(traincommandfile))
	
	strCmd = "{} detector train {} {} {} -dont_show -mjpeg_port 8090 -clear -gpus 0".format(\
		os.path.join(darkhome,'darknet.exe'), \
		datasFile, \
		configFile, \
		pretrained)
		
	with open(traincommandfile, 'w') as the_file:
		the_file.write('echo off\n')
		the_file.write(r'set "startTime=%time: =0%"' + '\n')
		the_file.write('echo on\n')
		the_file.write(strCmd + '\n')
		the_file.write('echo off\n')
		the_file.write(r'set "endTime=%time: =0%"' + '\n')
		the_file.write('echo "*****************************"' + '\n')
		the_file.write('echo Start:    %startTime%' + '\n')
		the_file.write('echo End:      %endTime%' + '\n')
		the_file.write('echo "*****************************"' + '\n')
		the_file.write('echo on\n')
	the_file.close
	

	print('\n--> ' + strCmd + '\n')
	
	print('-->or execute\n' + strCmd + '\n')
	
	
if __name__ == '__main__':
	steps = 0
		
	# read parameter
	steps += 1
	print("[Step {}] Read parameter.".format(steps) )
	namesFile, \
	datasFile, \
	configFile, \
	traincommandfile, \
	originalFolder, \
	weightsFolder, \
	xmllabelFolder, \
	imgFolder, \
	txtlabelFolder, \
	txtFile, \
	darkhome, \
	config, \
	pretrained, \
	batch, \
	subdivisions, \
	width, \
	height, \
	numclusters, \
	classList = ReadParameter(strParameterFile)
	
	#print('numclusters : ' + numclusters)
	#print(numclusters)
	
	# create folder if necessary
	steps += 1
	print("[Step {}] Create folder if necessary.".format(steps) )
	create_folder()
	
	# split original data
	steps += 1
	print("[Step {}] Split original data.".format(steps) )
	
	listB = glob.glob(os.path.join(originalFolder, 'B_*.bmp'))
	listS = glob.glob(os.path.join(originalFolder, 'S_*.bmp'))
	listH = glob.glob(os.path.join(originalFolder, 'H_*.bmp'))
	
	random.shuffle(listB)
	random.shuffle(listS)
	random.shuffle(listH)
	
	indices_for_splitting = [int(len(listB)*ratio[0]), int(len(listB)*(ratio[0]+ratio[1]))]
	train_B, val_B, test_B = np.split(listB, indices_for_splitting)
	indices_for_splitting = [int(len(listS)*ratio[0]), int(len(listS)*(ratio[0]+ratio[1]))]
	train_S, val_S, test_S = np.split(listS, indices_for_splitting)
	indices_for_splitting = [int(len(listH)*ratio[0]), int(len(listH)*(ratio[0]+ratio[1]))]
	train_H, val_H, test_H = np.split(listH, indices_for_splitting)
	
	train = train_B.tolist() + train_S.tolist() + train_H.tolist()
	val = val_B.tolist() + val_S.tolist() + val_H.tolist()
	test = test_B.tolist() + test_S.tolist() + test_H.tolist()
	
	# read xml template
	steps += 1
	print("[Step {}] Read xml template.".format(steps) )
	labelXML = minidom.parse(strLabelingXMLTemplate)
	
	# get yolo label txt x, y, w, h
	steps += 1
	print("[Step {}] Get yolo label txt x, y, w, h.".format(steps) )
	x, y, w, h = transferYolo( strLabelingXMLTemplate, int(width), int(height) )
	
	# copy image to train/validate/test and generate yolo label txt.
	steps += 1
	print("[Step {}] Copy image to train/validate/test and generate yolo label txt.".format(steps))
	for eachImgFile in train:
		copyimgandyololabeltext(eachImgFile, imgFolder['train'], x, y, w, h)
	for eachImgFile in val:
		copyimgandyololabeltext(eachImgFile, imgFolder['validate'], x, y, w, h)
	for eachImgFile in test:
		copyimgandyololabeltext(eachImgFile, imgFolder['test'], x, y, w, h)

	# generate names file
	steps += 1
	print("[Step {}] Generate names file.".format(steps) )
	create_names(namesFile, classList)
	
	# generate datas file
	steps += 1
	print("[Step {}] Generate data file.".format(steps) )
	create_data(datasFile, classList, txtFile, namesFile, weightsFolder)
	
	# generate txt files
	steps += 1
	print("[Step {}] Generate txt file.".format(steps) )
	for eachClass in folder_classes:
		create_txt(imgFolder[eachClass], txtFile[eachClass])
		
	# calculate anchors
	steps += 1
	print("[Step {}] Calculate anchors.".format(steps) )
	anchors = cauculate_anchors(darkhome, datasFile, numclusters, width, height)
		
	# generate config file
	steps += 1
	print("[Step {}] Generate config file.".format(steps) )
	create_config(darkhome, config, configFile, datasFile, classList, batch, subdivisions, width, height, anchors)

	# print command
	steps += 1
	print("[Step {}] Generate training command.".format(steps) )
	show_train_command(darkhome, configFile, datasFile, pretrained, traincommandfile)
