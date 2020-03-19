import sys
sys.dont_write_bytecode = True
import cv2
import lena
import cube


def main():
	"""
	Main Function
	"""
	# Read video
	cap = cv2.VideoCapture('/home/default/ENPM673/Project1/Updated Code/Video_dataset/Tag0.mp4')
	
	# Get refernce AR Tag image
	ref_img_path = "../reference_images/ref_marker.png"
	ref_img = cv2.imread(ref_img_path)

	# # Display Lena
	lena.showLena(cap,ref_img)
	
	# Display Cube
	# cube.showCube(cap,ref_img)


if __name__ == '__main__':
	main()