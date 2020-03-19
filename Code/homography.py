import numpy as np

def getPerspectiveTransform(pt_g, pt_p):
	"""
	Gets the perspective transform.

	:param      pt_g:  The ground truth coordinates 
	:type       pt_g:  Numpy Array
	:param      pt_p:  The plane coordinated
	:type       pt_p:  Numpy Array

	:returns:   Homography Matrix.
	:rtype:     Numpy array
	"""
	A = np.array([[-pt_g[0][0], -pt_g[0][1],- 1, 0, 0, 0, pt_g[0][0]*pt_p[0][0], pt_g[0][1]*pt_p[0][0], pt_p[0][0]],
	            [0, 0, 0, -pt_g[0][0], -pt_g[0][1], -1, pt_g[0][0]*pt_p[0][1], pt_g[0][1]*pt_p[0][1], pt_p[0][1]],
	            [-pt_g[1][0], -pt_g[1][1],- 1, 0, 0, 0, pt_g[1][0]*pt_p[1][0], pt_g[1][1]*pt_p[1][0], pt_p[1][0]],
	            [0, 0, 0, -pt_g[1][0], -pt_g[1][1], -1, pt_g[1][0]*pt_p[1][1], pt_g[1][1]*pt_p[1][1], pt_p[1][1]],
	            [-pt_g[2][0], -pt_g[2][1],- 1, 0, 0, 0, pt_g[2][0]*pt_p[2][0], pt_g[2][1]*pt_p[2][0], pt_p[2][0]],
	            [0, 0, 0, -pt_g[2][0], -pt_g[2][1], -1, pt_g[2][0]*pt_p[2][1], pt_g[2][1]*pt_p[2][1], pt_p[2][1]],
	            [-pt_g[3][0], -pt_g[3][1],- 1, 0, 0, 0, pt_g[3][0]*pt_p[3][0], pt_g[3][1]*pt_p[3][0], pt_p[3][0]],
	            [0, 0, 0, -pt_g[3][0], -pt_g[3][1], -1, pt_g[3][0]*pt_p[3][1], pt_g[3][1]*pt_p[3][1], pt_p[3][1]]])

	U,S,V = np.linalg.svd(A)
	S = np.append(np.diag(S),np.zeros((U.shape[1],1)),1)
	H_pred = np.reshape(np.transpose(V)[:,[8]],(3,3))
	return H_pred