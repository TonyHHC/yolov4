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
			os.path.join(images_path, "*.jpg")) + \
			glob.glob(os.path.join(images_path, "*.png")) + \
			glob.glob(os.path.join(images_path, "*.jpeg")) + \
			glob.glob(os.path.join(images_path, "*.bmp"))



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


	
def print_TX_test(image_name, detections):

	strOutput = image_name + ' : '
	for label, confidence, bbox in detections:
		strOutput += '({}#{})'.format(label, confidence)
		
	print(strOutput)

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
	while True:
		# loop asking for new image paths if no list is given
		if args.input:
			if index >= len(images):
				break
			image_name = images[index]
		else:
			image_name = input("Enter Image Path: ")
			
		image, detections = image_detection(image_name, network, class_names, class_colors, args.thresh)
		
		print_TX_test(image_name, detections)
        
		index += 1


if __name__ == "__main__":
	# unconmment next line for an example of batch processing
	# batch_detection_example()
	main()
