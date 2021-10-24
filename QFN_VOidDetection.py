
#Author: Abhiram Allipra
###########################################################################

# import the necessary packages
from __future__ import print_function
import numpy as np
import argparse
import cv2
import PIL
from PIL import Image
import math
from scipy import ndimage
import glob

from imutils import perspective
from imutils import contours
import imutils

###########################################################################

def main():
	#specify the image folder path
	path= "/Users/abhiram/Desktop/kopernickus/c23/"
	
	
	#path=openpathImg.filename1
	print('path passed')
	print ('Path:',path)
	image_list = []

	for filename in glob.glob(path+'*.png'): #change according to the file types
		im_original=Image.open(filename)
		image_list.append(im_original)
		img = np.array(im_original)
		img_shape=img.shape
		print("FILE:",filename)
		cv2.imshow("o1",img)
		cv2.waitKey()

		#iterating for each image
		shape_transform(img,img_shape,im_original)

def shape_transform(image,shape,im_original):
	
	#add argument
	image_threshold= 37
	
	#resizing 
	scale_percent = 50 # percent of original size
	width = int(image.shape[1] * scale_percent / 100)
	height = int(image.shape[0] * scale_percent / 100)
	dim = (width, height)

	#save = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	#cv2.imwrite(r'C:\Users\ALA5SI\Desktop\intermediate1.jpg',save)

	#Conversion to gray
	if len(image.shape)==3:
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		print('Converted to gray')
		output = gray.copy()

	else:
		output=image.copy()
	#im.show()
	ret,output = cv2.threshold(output,image_threshold,255,cv2.THRESH_BINARY)

	#scalng the image to get fitted 
	scale_percent = 50 # percent of original size
	#width = int(thresh1.shape[1] * scale_percent / 100)
	#height = int(thresh1.shape[0] * scale_percent / 100)
	dim = (width, height)
	#output = cv2.resize(thresh1, dim, interpolation = cv2.INTER_AREA)
	
	cv2.imshow("Opened Image",output)
	cv2.waitKey()
	cv2.destroyAllWindows()
	

	#Printing the dimensions
	#print('Original Dimensions : ',thresh1.shape)
	#print('Resized Dimensions : ',output.shape)

	cv2.imshow("Thresholded", output)
	cv2.waitKey()

	#currently specified shape method
	#if multiple shape recognition is required, please add another shape function

	circle_method(output,shape,im_original)

def circle_method(Output,shape,im_original):
	# detect circles in the image
	#########################################################################################################################
	
	#add argument
	circle_threshold= 1.67
	#default 1.67
	font=cv2.FONT_ITALIC
	circles = cv2.HoughCircles(Output, cv2.HOUGH_GRADIENT,circle_threshold, 100)

	# ensure at least some circles were found
	if circles is not None:
		
		print('Circles:',circles)

		# convert the (x, y) coordinates and radius of the circles to integers
		circles = np.round(circles[0, :]).astype("int")

		#Declaring the empty lists
		sample_listx= []
		sample_listy= []
	
		# loop over the (x, y) coordinates and radius of the circles
		for (x, y, r) in circles:
			# draw the circle in the output image, then draw a rectangle
			# corresponding to the center of the circle
			cv2.circle(Output, (x, y), r, (0, 255, 0), 4) 
			
			sample_listx = np.append (sample_listx,x)
			sample_listy = np.append (sample_listy,y)
			
			
			cv2.imshow("output_circles", Output)
			cv2.waitKey()

		cv2.destroyAllWindows()
		arrayOrder(sample_listx,sample_listy,Output,shape,im_original)	

def arrayOrder(listx,listy,output,shape,im_original):
	
	#creation of numpy arrays
	data = np.array([],dtype=float, ndmin=2)
	data1= np.array([listx[range(0,4)]])
	data2= np.array([listy[range(0,4)]])
	
	print('data1',data1)
	print('')
	print('data2',data2)
	
	#Create another array arr2 with size of arr1    
	
	box1 = np.append(data1,data2,axis=0)
	box2=np.array(box1)
	box=np.transpose(box2)

	print('Box',box)   


	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box	
	rect = order_points_old(box)
	
	# check to see if the new method should be used for
	# ordering the coordinates
	print('Ordered:',rect)
	
	# show the re-ordered coordinates
	print(rect.astype("int"))
	print("")
	
	# loop over the original points and draw them
	for (x,y) in rect:
		cv2.circle(output, (int(x), int(y)), 5, (0,255,0), -1)
	
	# show the image
	cv2.imshow("Image", output)
	cv2.waitKey(0)

	print('box:',box)
	#cv2.imwrite('C://Users/ALA5SI/Desktop/intermediate.jpg',output)

	#defining the vertices
	left=rect[0][0]
	print('1',left)
	left=left-30

	top= rect[0][1]
	print('2',top)
	top=top-30

	bottom=rect[2][0]
	print('3',bottom)
	bottom=bottom+30
	
	right=rect[2][1]
	print('4',right)
	right=right+30

	angleCheck(output,left,top,bottom,right,shape,im_original)
	#opening image again since crop is incompatible with opencv
	#later= Image.open('C://Users/ALA5SI/Desktop/intermediate2.jpg')

def order_points_old(pts):
	# initialize a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype="float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates

	return rect

def angleCheck(Img,Left,Top,Bottom,Right,shape,im_original):
	Left=int(Left)
	Top=int(Top)
	Bottom=int(Bottom)
	Right=int(Right)
	#drawing for angle detection
	cv2.rectangle(Img,(Left,Top),(Right,Bottom),(0,255,0),3)
	cv2.imshow('Lines for Angle Detection',Img)
	cv2.waitKey()
	cv2.destroyAllWindows()

	img_edges = cv2.Canny(Img, 50, 100, apertureSize=3)
	lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

	#creating a list to store the measured angles
	angles = []

	for [[x1, y1, x2, y2]] in lines:
		cv2.line(Img, (x1, y1), (x2, y2), (255, 0, 0), 3)
		angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
		angles.append(angle)

	cv2.imshow("Detected lines", Img)    
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	print('angles',angles)

	#measuring the deflected angle
	var_angle=angles[0]-90
	print('Deflected angle',var_angle)

	#checking if the detected angle is a straight line or if there is no tilt
	if var_angle== 180|0|-180|90|-90:
		print('No rotation required')

	else:
		img_rotated = ndimage.rotate(Img, var_angle)
		#cv2.imwrite(r'C:\Users\ALA5SI\Desktop\test_rotated.jpg', output)  
		print('Angle of Image rotated: ', var_angle)
	cv2.imwrite('C://Users/ALA5SI/Desktop/intermediate.jpg',Img)
	cropSection(Left,Top,Bottom,Right,shape,im_original)

def cropSection(Left1,Top1,Bottom1,Right1,shape,im_original):	
	#opening image again since crop is incompatible with opencv
	output= Image.open('C://Users/ALA5SI/Desktop/intermediate.jpg')
	#original=Image.open('C://Users/ALA5SI/Desktop/intermediate1.jpg')
	
	#crop
	crop_ref1=im_original.crop((Left1,Top1,Right1,Bottom1)) #croping  source image
	crop_ref=np.array(crop_ref1)
	#cv2.imwrite(r'C:\Users\ALA5SI\Desktop\intermediate1_crop.jpg',crop_ref)


	crop11 = output.crop((Left1,Top1,Right1,Bottom1)) 
	crop=np.array(crop11)

	cv2.imshow("Marked Boundary", crop)
	cv2.waitKey()
	cv2.destroyAllWindows()

	mask_modification(crop,crop_ref,shape)

def mask_modification(cropImg, Crop_ref,shape):


	#reading the mask file
	mask=cv2.imread(r"C:\Users\ALA5SI\Desktop\New\mask9.jpg",0)
	ret,thresh_mask=cv2.threshold(mask,40,255,cv2.THRESH_BINARY)
	
	
	w,h=cropImg.shape
	dim1 = (w, h)

	print(' dimension cropped image:',dim1)
	print(' dimension mask image:',thresh_mask.shape)

	cv2.imshow("Loaded Mask",thresh_mask)
	cv2.waitKey()
	cv2.destroyAllWindows()

	#print ("White area before:",cv2.countNonZero(thresh))

	#resizing the mask
	mask = cv2.resize(thresh_mask, dim1, interpolation = cv2.INTER_AREA)
	mask=np.transpose(mask)

	print('resized mask dimension:',mask.shape)
	print('Crop dimension:',cropImg.shape)
	#print('Crop dimension2:',modi_mask.shape)

	
	#counting the mask white area in the modified mask
	#ret,thresh_modimask = cv2.threshold(modi_mask,40,255,cv2.THRESH_BINARY)
	#print (" area Ref:",cv2.countNonZero(modimask))

	#cv2.imwrite(r'C:\Users\ALA5SI\Desktop\out_mask.jpg',thresh_modimask)
	
	#cv2.imshow("final_mask",modi_mask)
	cv2.waitKey()
	cv2.destroyAllWindows()

	
	merge_img(mask,Crop_ref,shape)

def merge_img(Msk,crop_ref,shape):
	
	alpha = 0.5

	#loading images
	src1 = Msk

	cv2.imwrite(r'C:\Users\ALA5SI\Desktop\referencecrop.jpg',crop_ref)
	#thresholding to see the voids
	ret,src2 = cv2.threshold(crop_ref,53,255,cv2.THRESH_BINARY)
	
	
	width = int(src2.shape[1])
	height = int(src2.shape[0])
	dim = (width, height)
	#src1 = cv2.resize(src0, dim, interpolation = cv2.INTER_AREA)

	#cv2.imwrite(r'C:\Users\ALA5SI\Desktop\maskresized.jpg',src1)
	cv2.imshow('Loaded mask',src1) #larger
	cv2.imshow('Cropped reference region',src2)
	cv2.waitKey()

	# [load]
	if src1 is None:
		print("Error loading src1")
		exit(-1)
	elif src2 is None:
		print("Error loading src2")
		exit(-1)

	print('Shape src1:',src1.shape)
	print('Shape src2:',src2.shape)
	# [blend_images]
	beta = (1.0 - alpha)
	dst = cv2.addWeighted(src1, alpha, src2, beta, 0.0)
	# [blend_images]
	# [display]
	cv2.imwrite(r'C:\Users\ALA5SI\Desktop\dst.jpg',dst)

	#img =cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
	ret,thresh1 = cv2.threshold(dst,142,255,cv2.THRESH_BINARY)
	cv2.imwrite('C://Users/ALA5SI/Desktop/outputimage.jpg',dst)


	#print('White pixel area1:',cv.countNonZero(img))
	
	cv2.imshow('Merged Picture', dst)
	cv2.imshow('Merged-Thresholded', thresh1 )
	cv2.imwrite(r'C:\Users\ALA5SI\Desktop\thresholded_Final.jpg',thresh1)
	cv2.waitKey(0)
	print('pixel area void:',cv2.countNonZero(thresh1))
	# [display]
	cv2.destroyAllWindows()
	#backtoLarge(thresh1,shape)

if __name__ == "__main__":
	main()
