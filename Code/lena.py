import cv2
import imutils
import numpy as np

import copy
import warp
import utils
import orientations as ort
import homography as hm


def showLena(cap,ref_img):
	"""
	Replace AR Tag with Lena.

	:param      cap:      The reader object of video
	:type       cap:      cv2.videoCapture()
	:param      ref_img:  The reference image
	:type       ref_img:  Image
	"""
	ref_gray = cv2.cvtColor(ref_img,cv2.COLOR_BGR2GRAY)

	# Get orientation of the AR Tag Reference
	ref_orientation_vector = ort.getOrientation(ref_gray, viz=False)
	ref_gray = np.pad(ref_gray,70, mode='constant', constant_values=255)
	ref_img2 = cv2.cvtColor(ref_gray,cv2.COLOR_GRAY2BGR)
	ref_corners = utils.getContourCorners(ref_gray, ref_img2)
	ref_corners = ref_corners-70
	# utils.visualizeCorners(ref_img, ref_corners, "ref_corners")

	while True:

		# Read Frames from VideoWriterObject
		ret,frame_img = cap.read()

		frame_gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
		frame_corners = utils.getContourCorners(frame_gray, frame_img)
		
		## Visialise the detected corners
		# utils.visualizeCorners(frame_img, frame_corners, "frame_corners")

		# Find Homography between Reference Frame and Video Frame
		# print("frame_corners:\n", frame_corners)
		# print("ref_corners:\n", ref_corners)
		H = hm.getPerspectiveTransform(frame_corners,ref_corners)
		# print("homography Matrix",H)
		maxWidth = ref_img.shape[1]
		maxHeight = ref_img.shape[0]
		warped = warp.myWarpPerspective(frame_gray, H, (maxWidth, maxHeight))
		# cv2.imshow("Warped", np.asarray(warped, dtype=np.uint8))

		# Mask the AR Tag from frame
		ret, warped_thresh = cv2.threshold(warped, 245, 255, cv2.THRESH_BINARY)
		kernel = np.ones((5,5),np.uint8)
		warped_thresh = cv2.erode(warped_thresh,kernel,iterations = 1)
		# cv2.imshow("Warped thresh", warped_thresh)

		# Get orientation of the AR Tag 
		frame_orientation_vector = ort.getOrientation(warped_thresh, viz=True)
		print("Reference Orientation:", ref_orientation_vector)
		print("Tag Orientation:", frame_orientation_vector)

		# Calculate the number of 90 degree rotations in the clockwise direction
		num_90_degs = ort.getRotation(ref_orientation_vector, frame_orientation_vector)
		# print("num_90_degs:", num_90_degs)


		# Find AR Tag ID
		warped_thresh_rotated = np.rot90(warped_thresh, k=num_90_degs)
		id_vec = utils.getTagID(warped_thresh_rotated, viz=True)
		print("Tag ID",id_vec)

		###########################################
		## 		ADD LENA 						##
		###########################################
		#Load Lena Image
		lena_img = '../reference_images/Lena.png' 
		lena_img = cv2.imread(lena_img,-1)
		lena_img = cv2.resize(lena_img,(200,200))
		lena_img = imutils.rotate(lena_img,-num_90_degs*90)
		lena_c = np.array([[0,0],[0,lena_img.shape[1]],[lena_img.shape[0],lena_img.shape[1]],[lena_img.shape[0],0]])
		# utils.visualizeCorners(lena_img,lena_c,"Lena Corners")
		
		# Add Lena Image to Frame
		warped = np.asarray(warped, dtype=np.uint8)
		w_c = cv2.cvtColor(warped,cv2.COLOR_GRAY2RGB)

		# print(w_c.shape)
		w_c[:,:] = lena_img
		# cv2.imshow("Lena Added:", w_c)

		# Unwarp the Lena Image onto the Frame
		unwarp = warp.myWarpPerspectiveSparse(w_c,np.linalg.inv(H),np.array([1080,1920,3]))
		dst = copy.deepcopy(frame_img)
		dst[unwarp!=0] = unwarp[unwarp!=0]
		cv2.imshow("Unwarped:", dst)
		cv2.imwrite("lena.png",dst)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()