import sys
sys.path.append('D:/Project/Tony/Python/yolov4/darknet/darknet/build/darknet/x64')

import argparse
import os
import glob
import random
import darknet
import time
import cv2
import numpy as np
import darknet
import re

listHistory = ['H','H','H']
strPrefix = '1d'
iTrust = 0.25

offsets_pct = {
	"None"     : (    0,     0),
	"Left"     : (-0.29,     0),
	"LeftDown" : (-0.29,  0.29),
	"Up"       : (    0, -0.29),
	"LeftUp"   : (-0.29, -0.29)
}
offset_x_addon = (0,-1,-2,-3,-4,-5)

strTempImgFilePath = 'D:/Temp/blank.png'
pixel_tgt = None
strOldImgName = ''

def parser():
	parser = argparse.ArgumentParser(description="YOLO Object Detection")
	parser.add_argument("--input", type=str, default="", help="image source. It can be a single image, a"
                        "txt with paths to them, or a folder. Image valid"
                        " formats are jpg, jpeg or png."
                        "If no input is given, ")
	parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")
	parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
	parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
	parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
	parser.add_argument("--save_labels", action='store_true',
                        help="save detections bbox for each image in yolo format")
	parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
	parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
	parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")
	return parser.parse_args()


def check_arguments_errors(args):
	assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
	if not os.path.exists(args.config_file):
		raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
	if not os.path.exists(args.weights):
		raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
	if not os.path.exists(args.data_file):
		raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
	if args.input and not os.path.exists(args.input):
		raise(ValueError("Invalid image path {}".format(os.path.abspath(args.input))))


def load_images(images_path):
	"""
	If image path is given, return it directly
	For txt file, read it and return each line as image path
	In other case, it's a folder, return a list with names of each
	bmp, jpg, jpeg and png file
	"""
	
	input_path_extension = images_path.split('.')[-1]
	if input_path_extension in ['jpg', 'jpeg', 'png', 'bmp']:
		return [images_path]
	elif input_path_extension == "txt":
		with open(images_path, "r") as f:
			return f.read().splitlines()
	else:
		return glob.glob(
			os.path.join(images_path, strPrefix+"*.jpg")) + \
			glob.glob(os.path.join(images_path, strPrefix+"*.png")) + \
			glob.glob(os.path.join(images_path, strPrefix+"*.jpeg")) + \
			glob.glob(os.path.join(images_path, strPrefix+"*.bmp"))



def image_detection(image_path, network, class_names, class_colors, thresh):
	# Darknet doesn't accept numpy images.
	# Create one with image we reuse for each detect
	width = darknet.network_width(network)
	height = darknet.network_height(network)
	darknet_image = darknet.make_image(width, height, 3)

	image = cv2.imread(image_path)
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image_resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)

	darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
	detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
	darknet.free_image(darknet_image)
	image = darknet.draw_boxes(detections, image_resized, class_colors)
	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections



def image_classification(image, network, class_names):
	width = darknet.network_width(network)
	height = darknet.network_height(network)
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image_resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
	darknet_image = darknet.make_image(width, height, 3)
	darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
	detections = darknet.predict_image(network, darknet_image)
	predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]
	darknet.free_image(darknet_image)
	return sorted(predictions, key=lambda x: -x[1])


def convert2relative(image, bbox):
	"""
	YOLO format use relative coordinates for annotation
	"""
	x, y, w, h = bbox
	height, width, _ = image.shape
	return x/width, y/height, w/width, h/height
	
def find_rightmost_box_x1_pos(detections):
	rLabel = 'na'
	x1_pos = -1
	rConfidence = 0
	for label, confidence, bbox in detections:
		if bbox[0] > x1_pos:
			x1_pos = bbox[0]
			rLabel = label
			rConfidence = confidence
			
	return rLabel, x1_pos, rConfidence

	
def print_TX_test(image_name, detections, moveKey, offset, eachAddon):

	global strOldImgName

	iWait = 0
	iFound = 0
	iMatch = 0

	token = re.split('_|\\.', image_name)
	strBuySell = token[3]
	iRightmost = int(token[4]) + offset[0]

	strOutput = image_name + ' : '
	
	label, x1_pos, confidence = find_rightmost_box_x1_pos(detections)
	
	if strBuySell[1] != 'H':
		strOutput += '<Wait >'
		iWait = 1
	else:
		strOutput += '<     >'
		
	if label[0] in ['B','S'] and x1_pos >= iRightmost and float(confidence) >= iTrust:
		strOutput += '<{}>'.format(label[0])
		iFound = 1
	else:
		strOutput += '< >'
		
	strOutput += '<{:8s}>'.format(moveKey)
	strOutput += '<{:3d}>'.format(eachAddon)
		
	if label[0] in listHistory and iFound == 1:
		strOutput += '<Match>'
		iMatch = 1
		listHistory[:] = ['H' for aa in listHistory[:]]
	else:
		strOutput += '<     >'
	
	if strOldImgName != image_name:
		listHistory.pop(0)
		listHistory.append(strBuySell[1])
		strOldImgName = image_name
	
	strOutput += '({}#{}#{})'.format(label, confidence, x1_pos)
	#print(strOutput)
	
	return iWait, iFound, iMatch, strOutput
	
def prepareWorkingImage(image_name, key, offset_x_addon):

	global pixel_tgt

	# prepare blank image
	image = cv2.imread(image_name)
	image_height, image_width, number_of_color_channels = image.shape
	color = (255,255,255) # opencv is BGR, not RGB
	if pixel_tgt is None :
		pixel_tgt = np.full((image_height, image_width, number_of_color_channels), color, dtype=np.uint8)
	else :
		pixel_tgt[:] = (255,255,255)

	offset = np.int_(np.multiply((image_width, image_height), offsets_pct[key])).tolist()
	offset[0] += offset_x_addon
	
	if offset[1] >= 0 :
		pixel_tgt[offset[1]:(image_height-1),0:(image_width+offset[0]-1)] = image[0:(image_height-offset[1]-1),(-offset[0]):(image_width-1)] # move left 150, down 0 or 150
	else :
		pixel_tgt[0:(image_height+offset[1]-1),0:(image_width+offset[0]-1)] = image[(-offset[1]):(image_height-1),(-offset[0]):(image_width-1)] # move up 150, down 0 or 150
		
	cv2.imwrite(strTempImgFilePath, pixel_tgt)

	image = None
	del image
	
	return strTempImgFilePath, offset
	

def main():
	args = parser()
	check_arguments_errors(args)

	random.seed(3)  # deterministic bbox colors
	network, class_names, class_colors = darknet.load_network(
		args.config_file,
		args.data_file,
		args.weights,
		batch_size=args.batch_size
	)

	images = load_images(args.input)

	index = 0
	iTotalImage = 0
	iTotalWait = 0
	iTotalFound = 0
	iTotalMatch = 0
	while True:
		# loop asking for new image paths if no list is given
		if args.input:
			if index >= len(images):
				break
			image_name = images[index]
		else:
			image_name = input("Enter Image Path: ")

		iTotalImage += 1
		
		iWait = 0
		iFound = 0
		iMatch = 0

		for moveKey in offsets_pct:
			strOutput = ''
			for eachAddon in offset_x_addon:
				tmpImgFile, offset = prepareWorkingImage(image_name, moveKey, eachAddon)
				image, detections = image_detection(tmpImgFile, network, class_names, class_colors, args.thresh)
		
				iWait, iFound, iMatch, strOutput = print_TX_test(image_name, detections, moveKey, offset, eachAddon)
			
				if iFound != 0:
					print(strOutput)
					strOutput = ''
					break;
			
			if iFound != 0:
				break;
				
		if strOutput != '':
			print(strOutput)
		
		iTotalWait += iWait
		iTotalFound += iFound
		iTotalMatch += iMatch
        
		index += 1
	
	print('\n')
	print('Total image  : {}\n'.format(iTotalImage))
	print('Total wait  : {}\n'.format(iTotalWait))
	print('Total found : {}\n'.format(iTotalFound))
	print('Total match : {}\n'.format(iTotalMatch))


if __name__ == "__main__":
	# unconmment next line for an example of batch processing
	# batch_detection_example()
	main()
