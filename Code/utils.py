import numpy as np
import cv2
import copy


def getContourCorners(img_gray, img):
	"""
	Gets the contour corners.

	:param      img_gray:  The grayscale image
	:type       img_gray:  Image
	:param      img:       The color image
	:type       img:       Image

	:returns:   The contour corners.
	:rtype:     List of corners
	"""

	ret,thresh = cv2.threshold(img_gray,230,255,0)
	image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	corners = cv2.approxPolyDP(contours[1],0.009 * cv2.arcLength(contours[1], True), True) 
	corners = np.squeeze(corners,axis=1)
	return corners


def visualizeCorners(img, corners, frame_name):
	"""
	Visualize the corners on Image

	:param      img:         The image
	:type       img:         Image
	:param      corners:     The corners
	:type       corners:     The list of (x,y) coords
	:param      frame_name:  The frame name to be displayed
	:type       frame_name:  String
	"""
	img_copy = copy.deepcopy(img)
	cnt = 0
	for i in corners:
		x,y = i.ravel()
		cnt+=1

		color = (50*cnt,50*cnt,0)
		cv2.circle(img_copy, (x,y), 10, color, -1)

	cv2.imshow(frame_name, img_copy)	



def getBitValue(block):
	"""
	Returns the bit value of block.

	:param      block:  The AR Tag Block
	:type       block:  Image

	:returns:   The bit value.
	:rtype:     int
	"""
	if np.mean(block) > 127:
		return 1
	return 0


def getTagID(img, viz=False):
	"""
	Returns the AR tag id.

	:param      img:  The image
	:type       img:  Image
	:param      viz:  Visualization zflag
	:type       viz:  boolean

	:returns:   The tag id.
	:rtype:     Numpy Array
	"""
	row_width = int(img.shape[0] / 8)
	col_width = int(img.shape[1] / 8)

	block_top_left = img[3*row_width:4*row_width, 3*col_width:4*col_width]
	block_top_right = img[3*row_width:4*row_width, 4*col_width:5*col_width]
	block_bottom_left = img[4*row_width:5*row_width, 3*col_width:4*col_width]
	block_bottom_right = img[4*row_width:5*row_width, 4*col_width:5*col_width]

	# if viz:
	# 	cv2.imshow("ID block_top_left", block_top_left)
	# 	cv2.imshow("ID block_top_right", block_top_right)
	# 	cv2.imshow("ID block_bottom_left", block_bottom_left)
	# 	cv2.imshow("ID block_bottom_right", block_bottom_right)

	# Order of bit values
	# _________
	# |_0_|_3_|
	# |_1_|_2_|

	id_vec = [getBitValue(block_top_left), getBitValue(block_bottom_left), getBitValue(block_bottom_right), getBitValue(block_top_right)]

	return id_vec


def updateBorders(img):
	"""
	Unpad the corners by 70 pixels

	:param      img:  The image
	:type       img:  Image

	:returns:   Unpadded IMage
	:rtype:     Image
	"""
	# img = img[img.sum(axis=1)!=255, :]
	# img = img[:, img.sum(axis=0)!=255]
	img = img[70:img.shape[0]-70,70:img.shape[1]-70]
	return img


def visualizeCube(img,base,top,frame_name):
	"""
	Visualize the Cube 

	:param      img:         The image
	:type       img:         Image
	:param      base:        The cube base coordinates
	:type       base:        Numpy array
	:param      top:         The cube top coordinated
	:type       top:         Top array
	:param      frame_name:  Frame Name
	:type       frame_name:  String
	"""
	cv2.polylines(img,[base],True,(0,0,255),5)
	cv2.polylines(img,[top],True,(0,255,0),5)
	for i in range(0,4):
		cv2.line(img,(base[i,0],base[i,1]),(top[i,0],top[i,1]),(255,0,0),5)
	cv2.imshow(frame_name,img)
