import glob, os
import os.path
import subprocess
import time
import re
from shutil import copyfile
import shutil
import cv2
from xml.dom import minidom
from os.path import basename
from tqdm import tqdm

#--------------------------------------------------------------------
strParameterFile = 'D:/Project/Tony/Python/yolov4/darknet/darknet/build/darknet/x64/tony_data_01/00_parameter.xml'
folder_classes = ('train', 'validate')
cfgs_total = {
	"yolov4": ["cfg/yolov4.cfg", "pretrained/yolov4.conv.137", '608_9'],
	"yolov4-tiny": ["cfg/yolov4-tiny.cfg", "pretrained/yolov4-tiny.conv.29", '416_6'],
}

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
	return tmpNode.childNodes[0].nodeValue

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
	
	darkhome = getXmlChildNodeValue(root, 'darkhome')
	config = getXmlChildNodeValue(root, 'config')
	pretrained = getXmlChildNodeValue(root, 'pretrained')
	batch = getXmlChildNodeValue(root, 'batch')
	subdivisions = getXmlChildNodeValue(root, 'subdivisions')
	width = getXmlChildNodeValue(root, 'width')
	height = getXmlChildNodeValue(root, 'height')
	numclusters = getXmlChildNodeValue(root, 'numclusters')
	
	return namesFile, datasFile, configFile, traincommandfile, weightsFolder, xmllabelFolder, imgFolder, txtlabelFolder, txtFile, darkhome, config, pretrained, batch, subdivisions, width, height, numclusters, classList
		
def transferYolo( xmlFilepath, imgFilepath, txtlabelFilepath, classList):

	print("--> processing {}".format(xmlFilepath))
	
	if(os.path.isfile(txtlabelFilepath)):
		os.remove(txtlabelFilepath)

	if(os.path.isfile(xmlFilepath)):
		img = cv2.imread(imgFilepath)
		imgShape = img.shape
		img_h = imgShape[0]
		img_w = imgShape[1]

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

		yoloFilename = txtlabelFilepath
		#print("writeing to {}".format(yoloFilename))

		with open(yoloFilename, 'a') as the_file:
			i = 0
			for className in labelName:
				if(className in classList):
					classID = classList[className]
					x = (labelXmin[i] + (labelXmax[i]-labelXmin[i])/2) * 1.0 / img_w
					y = (labelYmin[i] + (labelYmax[i]-labelYmin[i])/2) * 1.0 / img_h
					w = (labelXmax[i]-labelXmin[i]) * 1.0 / img_w
					h = (labelYmax[i]-labelYmin[i]) * 1.0 / img_h

					the_file.write(str(classID) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n')
					i += 1
		the_file.close()
		
def create_folder():
	for key in txtlabelFolder:
		folder = txtlabelFolder[key]
		print("--> create {}".format(folder))
		if not os.path.exists(folder):
			os.makedirs(folder)

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
		the_file.write(strCmd)
	the_file.close
	

	print('\n' + strCmd + '\n')
	
	
if __name__ == '__main__':
	steps = 0
		
	# read parameter
	steps += 1
	print("[Step {}] Read parameter.".format(steps) )
	#namesFile, datasFile, configFile, weightsFolder, xmllabelFolder, imgFolder, txtlabelFolder, txtFile, classList = ReadParameter(strParameterFile)
	namesFile, \
	datasFile, \
	configFile, \
	traincommandfile, \
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
	
	# convert
	for eachClass in folder_classes:
		steps += 1
		print("[Step {}] Convert {} labeled images from xml to yolo format.".format(steps, eachClass))
		for file in (os.listdir(imgFolder[eachClass])):
			filename, file_extension = os.path.splitext(file)
			file_extension = file_extension.lower()
	
			if(file_extension == ".jpg" or file_extension==".png" or file_extension==".jpeg" or file_extension==".bmp"):
				imgfile = os.path.join(imgFolder[eachClass], file)
				xmllabelfile = os.path.join(xmllabelFolder[eachClass] ,filename + ".xml")
				txtlabelfile = os.path.join(txtlabelFolder[eachClass] ,filename + ".txt")
		
				transferYolo( xmllabelfile, imgfile, txtlabelfile, classList)
	
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