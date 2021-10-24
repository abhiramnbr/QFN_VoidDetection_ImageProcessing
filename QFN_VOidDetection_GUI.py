'''
Author: Abhiram Allipra
contact: abhiramnbr@gmail.com
'''
###########################################################################

from __future__ import print_function
#import PyQt5
import tkinter as tk
from tkinter import *
from tkinter import filedialog,Text
from tkinter.filedialog import asksaveasfile

import imutils
from imutils import perspective
from imutils import contours

import time
import joblib
import pickle
import math
import PIL
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
import tkinter.messagebox as tkmb
import csv
import argparse

from scipy import ndimage
import glob

#global 
list_img=[] #list to store images
list_mask=[] #list to store masks
thresh_val=[]#list for threshold values

final_percent=[]
current_file=[]

pts = [] # for storing points
alpha = 0.5 # for merge function

# Lists to store the points (reference area) (optional)
top_left_corner=[]
bottom_right_corner=[]

#scale for resizing
scale_percent=50

#####################################################################
class GUI:

	def draw_roi(event, x, y, flags, param):
	 # :mouse callback function
	 img=list_img[len(list_img)-1]
	 img=cv2.imread(img)
	 img2 = img.copy()
	 if event == cv2.EVENT_LBUTTONDOWN: # Left click, select point
		 pts.append((x, y))  
 
	 if event == cv2.EVENT_RBUTTONDOWN: # Right click to cancel the last selected point
		 pts.pop()  
 
	 if event == cv2.EVENT_MBUTTONDOWN: # 
		 mask = np.zeros(img.shape, np.uint8)
		 points = np.array(pts, np.int32)
		 points = points.reshape((-1, 1, 2))

		 mask1 = cv2.polylines(mask, [points], True, (255, 255, 255), 2, cv2.LINE_AA)
		 mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255)) # for ROI
		 mask3 = cv2.fillPoly(mask.copy(), [points], (0, 255, 0)) # for displaying images on the desktop
 
		 show_image = cv2.addWeighted(src1=img, alpha=0.8, src2=mask3, beta=0.2, gamma=0)
 
		 #cv2.imshow("mask", mask2)
		 #cv2.imshow("show_img", show_image)
 
		 ROI = cv2.bitwise_and(mask2, img)
		 print('shape',ROI.shape)
		 #cv2.imshow("ROI", ROI)
		 cv2.waitKey(0)
 
	 if len(pts) > 0:
		# Draw the last point in pts
		 cv2.circle(img2, pts[-1], 3, (0, 0, 255), -1)
 
	 if len(pts) > 1:
		 for i in range(len(pts) - 1):
			 cv2.circle(img2, pts[i], 5, (0, 0, 255), -1) # x ,y is the coordinates of the mouse click place
			 cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)
	 
	 cv2.imshow('image', img2)

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
	
	def openPathFolder():
		filename1=filedialog.askdirectory()
		for filename in glob.iglob(filename1 + '**/*.jpg', recursive=True):
			list_img.append(filename)
		info_message = "Folder Selected: "
		# info message box
		tkmb.showinfo("Output", info_message + filename1)
		print('Selected Path:',list_img)

	def drawReference():
		if (len(list_img) & len(list_mask)):
			img= list_img[len(list_img)-1]
			img = cv2.imread(img)
			img1=img.copy()

			width = int(img.shape[1] * scale_percent / 100)
			height = int(img.shape[0] * scale_percent / 100)
			dim = (width, height)
			

			
			cv2.namedWindow('image')
			cv2.setMouseCallback('image', GUI.draw_roi)

			info_message= '''Click the left mouse button to select the point, right mouse click to delete the last selected point, Hit Spacebar to proceed'''
			tkmb.showinfo("Info", info_message)

			#print("[INFO] Click the left button: select the point, right click: delete the last selected point, click the middle button: determine the ROI area")
			#print("[INFO] Press ‘S’ to determine the selection area and save it")
			#print("[INFO] Press ESC to quit")

			while True:
				key = cv2.waitKey(1) & 0xFF
				if key == 27:
					break
				if key == ord(" "):
					#saved_data = {"ROI": pts}
					#joblib.dump(value=saved_data, filename="config.pkl")
					mask = np.zeros(img.shape, np.uint8)
					points = np.array(pts, np.int32)
					points = points.reshape((-1, 1, 2))

					mask1 = cv2.polylines(mask, [points], True, (255, 255, 255), 2, cv2.LINE_AA)
					mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255)) # for ROI
					mask3 = cv2.fillPoly(mask.copy(), [points], (0, 255, 0)) # for displaying images on the desktop

					
					width_mask = int(mask2.shape[1])
					height_mask = int(mask2.shape[0])
					dim_mask=(width_mask,height_mask)

					width = int(img.shape[1])
					height = int(img.shape[0])
					dim=(width,height)

					mask3=cv2.resize(mask3,dim,interpolation= cv2.INTER_AREA)

					print('image shape:',dim)
					print('mask shape:',dim_mask)

					img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
					mask2=cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)

					#show_image = cv2.addWeighted(src1=img, alpha=0.8, src2=mask3, beta=0.2, gamma=0)
			
					#cv2.imshow("mask", mask2)
					cv2.destroyAllWindows()
					#cv2.imshow("show_img", show_image)
					cv2.waitKey(0)

					mask2=cv2.resize(mask2,dim,interpolation= cv2.INTER_AREA)

					ROI = cv2.bitwise_and(mask2, img)
					#print('shape',ROI.shape)
					cv2.imshow("ROI", ROI)
					cv2.waitKey(0)

					ROI=np.array(ROI)
					#ROI1 = cv2.cvtColor(ROI,cv2.COLOR_BGR2GRAY)
					#print('shape2',ROI1.shape)
					ret,ROI1 = cv2.threshold(ROI,52,255,cv2.THRESH_BINARY)
					
					# [load]
					src1=list_mask[len(list_img)-1]
					src1 = cv2.imread(src1)
					src1=cv2.cvtColor(src1,cv2.COLOR_BGR2GRAY)
					ret, ref_thresh = cv2.threshold(src1,52,255,cv2.THRESH_BINARY)

					#resizing mask w.r.t loaded original image
					
					src1 = cv2.resize(src1, dim, interpolation = cv2.INTER_AREA)
					
					
					
					print('shape_mask:',src1.shape)
					src2 = ROI1
			#src2 = cv2.imread(r'C:\Users\ALA5SI\Desktop\crop.jpg')


					# [load]
					if src1 is None:
						print("Error loading mask")
						exit(-1)
					elif src2 is None:
						print("Error loading src2")
						exit(-1)


			# [blend_images]
					beta = (1.0 - alpha)
					dst = cv2.addWeighted(src1, alpha, src2, beta, 0.0)
					# [blend_images]
					# [display]
					#img =cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
					ret,thresh1 = cv2.threshold(dst,142,255,cv2.THRESH_BINARY)
					#cv2.imwrite('C://Users/ALA5SI/Desktop/outputimage.jpg',dst)

					cv2.destroyAllWindows()
					#print('White pixel area1:',cv2.countNonZero(img))
					sold=cv2.countNonZero(ref_thresh)
					voi=cv2.countNonZero(thresh1)

					print('Reference region Area:',sold )
					print('White pixel area2:',voi)

					tot=(voi/sold)*100
					print('Void Percentage %=',tot)
					#cv2.imshow('dst', dst)
					#cv2.imshow('img', thresh1 )
					cv2.waitKey(0)

					# [display]
					cv2.destroyAllWindows()
					yourData= list_img[0] +":"+"\t"+ str(tot)
					lab = Label(canvas,text=yourData)
					lab.pack()




		else:
			info_message= "Image/Mask Not Loaded"
			tkmb.showinfo("Warning!", info_message)

	def drawEpad(action, x, y, flags, *userdata):
		# Referencing global variables 
		global top_left_corner, bottom_right_corner
		# Mark the top left corner, when left mouse button is pressed
		if action == cv2.EVENT_LBUTTONDOWN:
			top_left_corner = [(x,y)]
			if len(top_left_corner) > 0:
				cv2.circle(image, top_left_corner[-1],3,(0,0,255), -1)
				#if len(top_left_corner) > 1:
			print ('TopLeft:',top_left_corner)
			# When left mouse button is released, mark bottom right corner
		elif action == cv2.EVENT_LBUTTONUP:
			bottom_right_corner = [(x,y)]   
			if len(bottom_right_corner) > 0:
				cv2.circle(image, bottom_right_corner[-1],3,(0,0,255), -1)
			print('BottomRight:',bottom_right_corner) 
			# Draw the rectangle
			cv2.rectangle(image, top_left_corner[0], bottom_right_corner[0], (0,255,0),2,8)
			cv2.imshow("Window",image)

	def openPathMask():
		filename2= filedialog.askopenfilename(initialdir="/",title='Select Mask Image',
											filetypes=([("Image files","*.jpg"),("Image files","*.png"),
											("Image files","*.tif"),("Image files","*.gif"),
											("Image files","*.bmp")]))    
		list_mask.append(filename2)
		info_message = "Mask Loaded: "
		# info message box
		tkmb.showinfo("Output", info_message + filename2)
		print('Selected File:',filename2)

	def select2():
		val=int(circ.get())
		thresh_val.append(val)
		print(thresh_val)

	def select():
		#threshView(val1, val2)
		#print(val1, val2)
		sel =  int(thresho.get())  
		#filewin.config(text = sel)
		print(sel)

		#val= int(float(sel))
		img=list_img[len(list_img)-1]
		img=cv2.imread(img)
		img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		if img is None:
				print ('Error opening image!')
				#print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
				return -1
		img2=img
		#img = imutils.resize(img, width=800,height=800)
		
		ret, thresh =cv2.threshold(img,sel,255,cv2.THRESH_BINARY)

			
		#cv2.imshow('Thresh',thresh)
		cv2.waitKey(0)
		
		img = cv2.medianBlur(thresh, 5)
		
		#cv2.imshow('OPened Img',img)
		#cv2.waitKey(0)
	
		w,h=img2.shape
		rows = w
		print('ROWS:',rows)
		#div=rows
		div=int(thresh_val[len(thresh_val)-1])
		print('Selected value:',div)
		circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, rows / div,
									param1=100, param2=30,
									minRadius=30, maxRadius=100)
			
			
		if circles is not None:
			print('circles:',circles)
			circles = np.uint16(np.around(circles))
			for i in circles[0, :]:
				center = (i[0], i[1])
				# circle center
				#cv2.circle(img, center, 1, (0, 100, 100), 3)
				# # circle outline
				radius = i[2]
				cv2.circle(img, center, radius, (255, 0, 255), 3)
		circles = np.round(circles[0, :]).astype("int")
		
		sample_listx=[]
		sample_listy=[]
		
		for (x, y, r) in circles:
			sample_listx = np.append(sample_listx,x)
			sample_listy = np.append(sample_listy,y)
		
		print('x:',sample_listx)
		print('y:',sample_listy)
		#cv2.imshow("detected circles", img)
		cv2.waitKey(0)
		print('length of list x:',len(sample_listx))
		if len(sample_listx)>= 4:
			GUI.arrayOrder(sample_listx,sample_listy,img)
		#active =False

	def prgmProceed():
		
		if (len(list_img) & len(list_mask)):
			filewin = Toplevel(root, bg="#808080",padx=100,pady=100)
			filewin.title('Threshold Window')
			
			#label for slider 2
			selection = "Circle Threshold"
			label = Label(filewin)
			label.config(text = selection)
			label.pack()
			#slider 2
			w2= Scale(filewin, from_=0, to=10,length=1000,variable = circ, tickinterval=1,orient=HORIZONTAL)
			w2.pack()
			button2 = Button(filewin, text="Submit Circle",padx=10,pady=5,fg="White",bg="#263D42",command= GUI.select2)
			button2.pack(anchor=CENTER)
			
			#label for slider 1
			selection = "Image Threshold"
			label = Label(filewin)
			label.config(text = selection)
			label.pack()
			#slider 1
			w1 = Scale(filewin, from_=0, to=255,variable = thresho,length =1000,tickinterval=10 ,orient=HORIZONTAL)
			w1.pack()
			button1 = Button(filewin, text="Submit Threshold",padx=10,pady=5,fg="White",bg="#263D42",command= GUI.select)
			button1.pack(anchor=CENTER)

		else:
			info_message= "Image/Mask Not Loaded"
			tkmb.showinfo("Warning!", info_message)

	def nextstep(w1):
		image_threshold=int(w1)
		#circle_threshold=w2.get()
		print('values:',image_threshold)

	def openPathImg():
		filename = filedialog.askopenfilename(initialdir="/", title="Select an Image", filetypes=(("jpg files", "*.jpg"),("png files","*.png"),("all files", "*.*")))
		#filename=filedialog.askdirectory()
		list_img.append(filename)
		info_message = "Image Loaded: "
		# info message box
		tkmb.showinfo("Output", info_message + filename)
		print('Selected Path:',filename)
		return filename
	
	def imgPreview():
		if (len(list_img)):
			img=list_img[len(list_img)-1]
			img=cv2.imread(img)
			img = imutils.resize(img, width=800,height=800)

			mask=list_mask[len(list_mask)-1]
			mask=cv2.imread(mask)
			img = imutils.resize(img, width=800,height=800)

			cv2.imshow('Preview Image',img)
			cv2.imshow('Preview Mask',mask)
			cv2.waitKey(0)
			#img = cv2.imread("C://Users/ALA5SI/Desktop/1.jpg")
			#img = imutils.resize(img, width=800,height=800)
			#img=np.array(img)


			#Rearrang the color channel
			#b,g,r = cv2.split(img)
			#img = cv2.merge((r,g,b))

			# Convert the Image object into a TkPhoto object
			#im = Image.fromarray(img)
			#imgtk = ImageTk.PhotoImage(image=img) 
			
			# Put it in the display window
			#tk.Label(root, image=imgtk).pack()
		else:
			info_message= "No image to Preview"
			tkmb.showinfo("Warning!", info_message)

	def arrayOrder(listx,listy,output):
		
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
		rect = GUI.order_points_old(box)
		
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
		#cv2.imshow("Image", output)
		cv2.waitKey(0)

		#print('box:',box)
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



		GUI.angleCheck(output,left,top,bottom,right)
		#opening image again since crop is incompatible with opencv
		#later= Image.open('C://Users/ALA5SI/Desktop/intermediate2.jpg')

	def angleCheck(Img,Left,Top,Bottom,Right):
		Left=int(Left)
		Top=int(Top)
		Bottom=int(Bottom)
		Right=int(Right)
		#drawing for angle detection
		cv2.rectangle(Img,(Left,Top),(Right,Bottom),(0,255,0),3)
		#cv2.imshow('Lines for Angle Detection',Img)
		
		cv2.destroyAllWindows()

		img_edges = cv2.Canny(Img, 50, 100, apertureSize=3)
		lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

		#creating a list to store the measured angles
		angles = []

		for [[x1, y1, x2, y2]] in lines:
			cv2.line(Img, (x1, y1), (x2, y2), (255, 0, 0), 3)
			angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
			angles.append(angle)

		#cv2.imshow("Detected lines", Img)    
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
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
		GUI.cropSection(Left,Top,Bottom,Right)

	def cropSection(Left1,Top1,Bottom1,Right1):	
		#opening image again since crop is incompatible with opencv
		
		output= Image.open('C://Users/ALA5SI/Desktop/intermediate.jpg')
		#original=Image.open('C://Users/ALA5SI/Desktop/intermediate1.jpg')
		
		#crop
		file=list_img[len(list_img)-1]
		im_original=Image.open(file)
		
		crop_ref1=im_original.crop((Left1,Top1,Right1,Bottom1)) #croping  source image
		crop_ref=np.array(crop_ref1)
		
		#cv2.imwrite(r'C:\Users\ALA5SI\Desktop\intermediate1_crop.jpg',crop_ref)


		crop11 = output.crop((Left1,Top1,Right1,Bottom1)) 
		crop=np.array(crop11)

		#cv2.imshow("Marked Boundary", crop)
		#cv2.waitKey()
		cv2.destroyAllWindows()

		GUI.mask_modification(crop,crop_ref)

	def mask_modification(cropImg, Crop_ref):


		#reading the mask file
		mask=cv2.imread(list_mask[len(list_mask)-1],0)
		ret,thresh_mask=cv2.threshold(mask,40,255,cv2.THRESH_BINARY)
		
		
		w,h=cropImg.shape
		dim1 = (w, h)

		print(' dimension cropped image:',dim1)
		print(' dimension mask image:',thresh_mask.shape)

		#cv2.imshow("Loaded Mask",thresh_mask)
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

		
		GUI.merge_img(mask,Crop_ref)

	def merge_img(Msk,crop_ref):
		
		alpha = 0.5

		#loading images
		src1 = Msk

		#cv2.imwrite(r'C:\Users\ALA5SI\Desktop\referencecrop.jpg',crop_ref)
		#thresholding to see the voids
		ret,src2 = cv2.threshold(crop_ref,53,255,cv2.THRESH_BINARY)
		
		
		width = int(src2.shape[1])
		height = int(src2.shape[0])
		dim = (width, height)
		#src1 = cv2.resize(src0, dim, interpolation = cv2.INTER_AREA)

		#cv2.imwrite(r'C:\Users\ALA5SI\Desktop\maskresized.jpg',src1)
		#cv2.imshow('Loaded mask',src1) #larger
		#cv2.imshow('Cropped reference region',src2)
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
		
		#cv2.imshow('Merged Picture', dst)
		#cv2.imshow('Merged-Thresholded', thresh1 )
		cv2.imwrite(r'C:\Users\ALA5SI\Desktop\thresholded_Final.jpg',thresh1)
		cv2.waitKey(0)

		Void=cv2.countNonZero(thresh1)
		print('Void Pixel Area:',Void)

		ret,Msk = cv2.threshold(Msk,53,255,cv2.THRESH_BINARY)
		Refer=cv2.countNonZero(Msk)
		print('Reference Pixel Area:',Refer)
		voidPercentage = (Void/Refer)*100
		print('Void Area Percentage:',voidPercentage)
		# [display]
		cv2.destroyAllWindows()

		yourData= list_img[len(list_img)-1] +":"+"\t"+ str(voidPercentage)
		lab = Label(canvas,text=yourData)
		lab.pack()
		
		final_percent.append(voidPercentage)

	def donothing():

		'''
		filewin = Toplevel(root)
		filewin.geometry('100x100')

		button = Button(filewin, text="Do nothing button")
		button.pack()
		'''
		Flag=True

	def readMe():
		filewin = Toplevel(root)
		info_message = (''' 
		Hello. 
		This reading is intended to guide you through the entire process of this software.
		
		1. Input the image 
		2. Input the mask
		3. Hit Proceed Button
		4. Select the 'Circle Threshold' as a value 3,4 or 5 (Usually)
		5. Hit the Submit Circle Button begore Proceeding to next section
		6. Now select the threshold value for the image. This is usually between 36 to 44 and can change with components.
		   Now hit the 'Submit Threshold' and you can see the void percentage and the corresponding file name in the main window.
		   If by any chance, the program doesnot respond after hitting the 'Submit Threshold' button. Try using other combinations of
		   Circle threshold and Image Threshold.
		7. You can now export the result from the 'File' menu in the main window and selecting the 'Save' option.
		8. One can load the next image for evaluation by adding the iamge and mask again and repeating the procedure done before.

		Thank you.
		    ''')
		tk.Label(filewin, text=info_message).grid() 
		print()
	
	def contactMe():
		filewin = Toplevel(root)
		info_message = ('''
		Hello, seems like you have an issue.
		contact me on abhiramnbr@gmail.com
		Thank you. 
		    ''')
		tk.Label(filewin, text=info_message).grid() 
		print()

	def stop():
		"""Stop scanning by setting the global flag to False."""
		global running
		running = False

	def saveCSV():
		files = [('All Files', '*.*'), 
             ('Excel File', '*.xls'),
             ('Text Document', '*.txt'),('Excel File','*.csv')]
		file = filedialog.asksaveasfile(filetypes = files, defaultextension = files)
		file=file.name
		print('NOW',file)
		
		header = ['File + Void Percent']
		with open(file, 'w') as f:
			
			writer = csv.writer(f,delimiter=',')
			writer.writerow(header)
			#writer.writerow(header)
			writer.writerow(list_img)
			writer.writerow(final_percent) 
		
######################################################
root = tk.Tk()
root.title("Reference Region Analyzer")

thresho = DoubleVar()
circ= IntVar()

menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Open Folder", command=GUI.openPathFolder)
filemenu.add_command(label="Open Image", command=GUI.openPathImg)
filemenu.add_command(label="Open Mask", command=GUI.openPathMask)
filemenu.add_separator()
filemenu.add_command(label="Save", command=GUI.saveCSV)

filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)

#####################################################

menubar.add_cascade(label="File", menu=filemenu)
editmenu = Menu(menubar, tearoff=0)
editmenu.add_separator()


#####################################################


helpmenu = Menu(menubar, tearoff=0)
helpmenu.add_command(label="About", command=GUI.readMe)
helpmenu.add_command(label="Contact Support", command=GUI.contactMe)
menubar.add_cascade(label="Help", menu=helpmenu)

#####################################################

#gui specifications
canvas= tk.Canvas(root,height=100,width=100,bg="#808080")
canvas.pack()

#frame= tk.Frame(root,bg="white")
#frame.place(relwidth=0.8,relheight=0.8,relx=0.1, rely=0.02)

T = Text(canvas,height = 10, width = 80)
T.pack()

#########################################################################

#button_Preview
imgPreview=tk.Button(root,text="Preview",padx=10,
					pady=5,fg="white",bg="#263D42", command=GUI.imgPreview)
imgPreview.pack(anchor=CENTER)

#button_DrawReference
draw=tk.Button(root,text="Draw Reference",padx=10,
					pady=5,fg="white",bg="#263D42",command=GUI.drawReference)
draw.pack(anchor=CENTER)

#button_Proceed
runAnalysis=tk.Button(root,text="Proceed",padx=10,
					pady=5,fg="White",bg="#263D42",command=GUI.prgmProceed)
runAnalysis.pack(anchor=CENTER)

##########################################################################

root.config(menu=menubar)
#root.bind_all("<Control-o>", GUI.openPathImg)

#root.bind_all("<Control-m>", GUI.openPathMask)

root.mainloop()





