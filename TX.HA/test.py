import glob, os
import os.path
import numpy as np
import cv2

image_height = 512
image_width = 512
number_of_color_channels = 3
color = (255,255,255) # opencv is BGR, not RGB

pixel_tgt = None

offsets = {
	"None"     : (   0,    0),
	"Left"     : (-150,    0),
	"LeftDown" : (-150,  150),
	"Up"       : (   0, -150),
	"LeftUp"   : (-150, -150)
}

offsets_pct = {
	"None"     : (   0,    0),
	"Left"     : (-0.29,    0),
	"LeftDown" : (-0.29,  0.29),
	"Up"       : (   0, -0.29),
	"LeftUp"   : (-0.29, -0.29)
}

def main():

	for key in offsets_pct:
		print('{}:{}'.format(key, offsets_pct[key]))

	#offset = (0, 0)
	#offset = (-150, 0) # move left 150
	#offset = (0, 150) # move down 150
	#offset = (0, -150) # move up 150
	#offset = (-150, 150) # move left 150, down 150
	#offset = (-150, -150) # move left 150, up 150
	#offset = offsets['Left']
	offset = np.int_(np.multiply((image_width, image_height), offsets_pct['LeftDown'])).tolist()
	print(offset)

	image = cv2.imread('D:/Temp/60n_202112151045_BH_436.png')
	print(image.shape)
	
	global pixel_tgt
	if pixel_tgt is None:
		pixel_tgt = np.full((image_height, image_width, number_of_color_channels), color, dtype=np.uint8)

	#pixel_tgt[:] = image[:]
	#pixel_tgt[:,0:361] = image[:,150:511] # move left 150
	#pixel_tgt[150:511,:] = image[0:361,:] # move down 150
	#pixel_tgt[150:511,0:361] = image[0:361,150:511] # move left 150, down 150
	#pixel_tgt[0:361,:] = image[150:511,:] # move up 150
	#pixel_tgt[0:361,0:361] = image[150:511,150:511] # move left 150,  up 150

	if offset[1] >= 0 :
		pixel_tgt[offset[1]:(image_height-1),0:(image_width+offset[0]-1)] = image[0:(image_height-offset[1]-1),(-offset[0]):(image_width-1)] # move left 150, down 0 or 150
	else :
		pixel_tgt[0:(image_height+offset[1]-1),0:(image_width+offset[0]-1)] = image[(-offset[1]):(image_height-1),(-offset[0]):(image_width-1)] # move up 150, down 0 or 150

	#pixel_tgt[:, 0:(image_width+offset[0]-1)] = image[:, (-offset[0]):(image_width-1)] # move left 150
	#pixel_tgt[offset[1]:(image_height-1),0:(image_width+offset[0]-1)] = image[0:(image_height-offset[1]-1),(-offset[0]):(image_width-1)] # move left 150, down 150]
	#pixel_tgt[0:(image_height+offset[1]-1),0:(image_width+offset[0]-1)] = image[(-offset[1]):(image_height-1),(-offset[0]):(image_width-1)] # move up 150, down 150

	cv2.imwrite('D:/Temp/blank512.png', pixel_tgt)

	image = None
	del image


if __name__ == "__main__":
	# unconmment next line for an example of batch processing
	# batch_detection_example()
	main()