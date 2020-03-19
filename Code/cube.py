import cv2
import imutils
import numpy as np

import copy
import warp
import utils
import orientations as ort
import homography as hm


def showCube(cap,ref_img):
	"""
	Shows the cube.

	:param      cap:      The cap object of video
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


		# Find AR Tag ID
		warped_thresh_rotated = np.rot90(warped_thresh, k=num_90_degs)
		id_vec = utils.getTagID(warped_thresh_rotated, viz=True)
		print("Tag ID",id_vec)

		################################################################## 
		######  				3D CUBE								######
		################################################################## 

		# Given Intrinsic Parameters
		K =np.transpose([[1406.08415449821,0,0],
			[2.20679787308599, 1417.99930662800,0],
			[1014.13643417416, 566.347754321696,1]])

		# Homography matrix for mapping world->camera
		h = np.linalg.inv(H)
		# Decomposing H into Rotation and Translation Components
		RT = np.matmul(np.linalg.inv(K),h)
		r1 = RT[:,0].reshape(3,1)
		r2 = RT[:,1].reshape(3,1)
		t = RT[:,2].reshape(3,1)
		r3 = np.cross(RT[:,0],RT[:,1]).reshape(3,1)

		R = np.concatenate((r1,r2),axis=1)
		R = np.concatenate((R,r3),axis=1)
		R = np.concatenate((R,t),axis=1)
		# print("R|t: \n",R)

		# Cube World Coordinates
		cube_w = np.array([[0,0,-200,1],
						[200,0,-200,1],
						[200,200,-200,1],
						[0,200,-200,1]])

		cube_c = []
		for i in cube_w:
			cw = np.matmul(np.matmul(K,R),i)
			cw = cw/cw[2]
			cube_c.append(cw)
		cube_c = np.array(cube_c).astype(np.int0)
		# print("Cube Camera \n:",cube_c[:,0:2])

		utils.visualizeCube(frame_img,frame_corners,cube_c[:,0:2],"Cube")
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()