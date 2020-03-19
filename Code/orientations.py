import numpy as np
import cv2
import copy
import utils

def getOrientation(img, viz=False):
	"""
	Gets the orientation.

	:param      img:  The image
	:type       img:  Image
	:param      viz:  Visualization Flag
	:type       viz:  boolean

	:returns:   The orientation vector.
	:rtype:     Numpy Array
	"""
	row_width = int(img.shape[0] / 8)
	col_width = int(img.shape[1] / 8)

	block_top_left = img[2*row_width:3*row_width, 2*col_width:3*col_width]
	block_top_right = img[2*row_width:3*row_width, 5*col_width:6*col_width]
	block_bottom_left = img[5*row_width:6*row_width, 2*col_width:3*col_width]
	block_bottom_right = img[5*row_width:6*row_width, 5*col_width:6*col_width]

	# if viz:
	# 	cv2.imshow("block_top_left", block_top_left)
	# 	cv2.imshow("block_top_right", block_top_right)
	# 	cv2.imshow("block_bottom_left", block_bottom_left)
	# 	cv2.imshow("block_bottom_right", block_bottom_right)

	# Order of bit values
	# ___________
	# |_0_|__|_3_|
	# |___|__|___|
	# |_1_|__|_2_|

	orientation_vec = [utils.getBitValue(block_top_left), utils.getBitValue(block_bottom_left), utils.getBitValue(block_bottom_right), utils.getBitValue(block_top_right)]

	return orientation_vec


def getRotation(ref_orientation_vector, frame_orientation_vector):
	"""
	Gets the rotation angle.

	:param      ref_orientation_vector:    The reference orientation vector
	:type       ref_orientation_vector:    Numpy Array
	:param      frame_orientation_vector:  The frame orientation vector
	:type       frame_orientation_vector:  Numpy Array

	:returns:   The rotation.
	:rtype:     Numpy Matrix
	"""

	if ref_orientation_vector.count(1) != 1:
		print("ERROR: Incorrect reference orientation")
		return 0
	elif frame_orientation_vector.count(1) != 1:
		print("ERROR: Incorrect frame orientation")
		return 0
	else:
		num_rots = 0
		while frame_orientation_vector != ref_orientation_vector:
		    frame_orientation_vector = frame_orientation_vector[-1:] + frame_orientation_vector[:-1]
		    num_rots += 1

		return num_rots