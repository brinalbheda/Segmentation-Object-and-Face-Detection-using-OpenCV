#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 12:56:11 2018

@author: brinalbheda
"""

import os
import numpy as np
import cv2
import random

os.chdir("/Users/brinalbheda/Desktop/OpenCV/Ex_Files_OpenCV_Python_Dev/Exercise Files/")


############ 02_01 ############

img = cv2.imread("opencv-logo.png")

#shows the window which will display the image
cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
cv2.imshow("Image",img)

#image will hang for specified ms or indefinitely is 0
cv2.waitKey(0)

#to convert the image into .jpg format
cv2.imwrite("output.jpg",img)


############ 02_02 ############

img = cv2.imread("opencv-logo.png", 1)

#prints the whole array of pixels 
print(img)
#leave a blank line
print('\n')

#show the image type --> numpy array
print(type(img))

#to find length ie number of horizontal rows in the image
print(len(img))
#to find number of vertical columns
print(len(img[0]))
#to find number of channels ie RGBalpha channels
print(len(img[0][0]))

#defining the above three factors
print(img.shape)

#determining the number of pixels ie 2^8(max value of image to determine data structure)
print(img.dtype)

#accessing the image pixels ie 10th row and 5th column pixel
print(img[10,5])

#to display any one channel image
print(img[:,:,0])

#to determine total number of pixels in the image
print(img.size)


############ 02_03 ############

#learn to make numpy arrays to deal with images
#to determine the shape and size of image
#black=0 & white=255
black = np.zeros([150,200,1],'uint8')
cv2.imshow("Black",black)
print(black[0,0,:])

ones = np.ones([150,200,3],'uint8')
cv2.imshow("Ones",ones)
print(ones[0,0,:])

#white image
white = np.ones([150,200,3],'uint16')
white *= (2**16-1)
cv2.imshow("White",white)
print(white[0,0,:])

#make a color image
color = ones.copy()
#BGR format
color[:,:] = (255,0,0)
cv2.imshow("Blue",color)
print(color[0,0,:])

cv2.waitKey(0)


############ 02_04 ############

color = cv2.imread("butterfly.jpg",1)
cv2.imshow("Image",color)

#moves window to the top left
cv2.moveWindow("Image",0,0)
print(color.shape)
height,width,channels = color.shape

b,g,r =cv2.split(color)
#split colors to individual channels
#create empty matrix
rgb_split = np.empty([height,width*3,3],'uint8')

#displaying individual sections by using splices for each channels
rgb_split[:,0:width] = cv2.merge([b,b,b])
rgb_split[:,width:width*2] = cv2.merge([g,g,g])
rgb_split[:,width*2:width*3] = cv2.merge([r,r,r])

cv2.imshow("Channels",rgb_split)
cv2.moveWindow("Channels",0,height)


#hue-saturation-value space
hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv)
hsv_split = np.concatenate((h,s,v),axis=1)
cv2.imshow("Split HSV",hsv_split)


cv2.waitKey(0)
cv2.destroyAllWindows()

 
############ 02_05 ############

color = cv2.imread("butterfly.jpg",1)

#convert into greyscale image
gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
cv2.imwrite("gray.jpg",gray)
cv2.imshow("Gray",gray)

#add an additional channel to the image
b = color[:,:,0]
g = color[:,:,1]
r = color[:,:,2]

#combine r,g,b,a into single image
rgba = cv2.merge((b,g,r,g))
cv2.imwrite("rgba.png",rgba)
#cv2.imshow("RGBA",rgba)


############ 02_06 ############

#deniosing the image
#Gaussian blur, dilation and erosion

image = cv2.imread("thresh.jpg")
cv2.imshow("Original",image)

#Gaussian --> average effect
blur = cv2.GaussianBlur(image, (5,55),0)
cv2.imshow("Blur",blur)

#defining and sliding the kernel over the pixels
kernel = np.ones((5,5),'uint8')

#Dilation --> black(background to white pixels
dilate = cv2.dilate(image,kernel,iterations=1)
#Erosion --> white to black pixels
erode = cv2.erode(image,kernel,iterations=1)

cv2.imshow("Dilate",dilate)
cv2.imshow("Erode",erode)

cv2.waitKey(0)
cv2.destroyAllWindows()


############ 02_07 ############

img = cv2.imread("players.jpg",1)

# Scale
img_half = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
img_stretch = cv2.resize(img, (600,600))
img_stretch_near = cv2.resize(img, (600,600), interpolation=cv2.INTER_NEAREST)

cv2.imshow("Half",img_half)
cv2.imshow("Stretch",img_stretch)
cv2.imshow("Stretch near",img_stretch_near)

# Rotation
M = cv2.getRotationMatrix2D((0,0), -30, 1)
rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
cv2.imshow("Rotated",rotated)

cv2.waitKey(0)
cv2.destroyAllWindows()


############ 02_08 ############


#video capture along with the frames
cap = cv2.VideoCapture(0)

while(True):
	ret, frame = cap.read()

	frame = cv2.resize(frame, (0,0), fx=0.5,fy=0.5)
	cv2.imshow("Frame",frame)

	ch = cv2.waitKey(1)
	if ch & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()


############ 02_09 ############

cap = cv2.VideoCapture(0)

#circle parameters
color = (0,255,0)
line_width = 3
radius = 100
point = (0,0)

def click(event, x, y, flags, param):
	global point, pressed
	if event == cv2.EVENT_LBUTTONDOWN:
		print("Pressed",x,y)
		point = (x,y)

#create circle on the live video stream when mouse clicks
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame",click)

while(True):
	ret, frame = cap.read()

	frame = cv2.resize(frame, (0,0), fx=0.5,fy=0.5)
	cv2.circle(frame, point, radius, color, line_width)
	cv2.imshow("Frame",frame)

	ch = cv2.waitKey(1)
	if ch & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()


############ 02_10 ############

# Global variables
canvas = np.ones([500,500,3],'uint8')*255
radius = 3
color = (0,255,0)
#to print circles continuously when cursor is pressed and moved
pressed = False

# click callback
def click(event, x, y, flags, param):
	global canvas, pressed
	if event == cv2.EVENT_LBUTTONDOWN:
		pressed = True
		cv2.circle(canvas,(x,y),radius,color,-1)
	elif event == cv2.EVENT_MOUSEMOVE and pressed == True:
		cv2.circle(canvas,(x,y),radius,color,-1)
	elif event == cv2.EVENT_LBUTTONUP:
		pressed = False

# window initialization and callback assignment
cv2.namedWindow("canvas")
cv2.setMouseCallback("canvas", click)

# Forever draw loop
while True:

	cv2.imshow("canvas",canvas)

	# key capture every 1ms
     #press the respective letter key to change the color
	ch = cv2.waitKey(1)
	if ch & 0xFF == ord('q'):
		break
	elif ch & 0xFF == ord('b'):
		color = (255,0,0)
	elif ch & 0xFF == ord('g'):
		color = (0,255,0)
	
cv2.destroyAllWindows()




############ 03_02 ############

#segmentation & object detection --> simple thresholding -> value based, adaptive thresholding -> local brightness variances, edge detection -> separate object boundaries, gausssian blurs -> reduce image noise, dilation and erosion -> expand/contract segmented areas

#Segmentation and binary images

#simple thresholding -> turning gray into black(background) and white(objects)
bw = cv2.imread('detect_blob.png', 0)
height, width = bw.shape[0:2]
cv2.imshow("Original BW",bw)

binary = np.zeros([height,width,1],'uint8')

thresh = 85
#all above threshold are made 1 ie 255 else less than threshold made 0
for row in range(0,height):
	for col in range(0, width):
		if bw[row][col]>thresh:
			binary[row][col]=255

cv2.imshow("Slow Binary",binary)

#using opencv library to make binary for above for loop --> same result
ret, thresh = cv2.threshold(bw,thresh,255,cv2.THRESH_BINARY)
cv2.imshow("CV Threshold",thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()


############ 03_03 ############

img = cv2.imread('sudoku.png',0)
cv2.imshow("Original",img)

ret, thresh_basic = cv2.threshold(img,70,255,cv2.THRESH_BINARY)
cv2.imshow("Basic Binary",thresh_basic)

#adaptive thresholding for different shades or uneven lighting
#115 is the value of how far the thresholding will act
thres_adapt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
cv2.imshow("Adaptive Threshold",thres_adapt)

cv2.waitKey(0)
cv2.destroyAllWindows()


############ 03_04 ############

#skin detection -> different shades of colors of face
img = cv2.imread('faces.jpeg',1)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h = hsv[:,:,0]  #hue -> red represented as 0 or 255 and blue or green as gray values
s = hsv[:,:,1]  #saturation
v = hsv[:,:,2]  #value

hsv_split = np.concatenate((h,s,v), axis=1)
cv2.imshow("Split HSV",hsv_split)

#above 40 turns white
ret, min_sat = cv2.threshold(s,40,255, cv2.THRESH_BINARY)
cv2.imshow("Sat Filter",min_sat)

#below 15 turns white
ret, max_hue = cv2.threshold(h,15, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Hue Filter",max_hue)

#combine sat and hue -> using multiple filters for thresholding give better results
final = cv2.bitwise_and(min_sat,max_hue)
cv2.imshow("Final",final)
cv2.imshow("Original",img)

cv2.waitKey(0)
cv2.destroyAllWindows()


############ 03_06 ############

#contours -> for object detection
#iterative converge algorithm until object spotted ie perimeter of object

img = cv2.imread('detect_blob.png',1)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
cv2.imshow("Binary", thresh)

#contours command, hierarchy -> parent child relationship of all contours
_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img2 = img.copy()
index = -1
thickness = 4
color = (255, 0, 255)

cv2.drawContours(img2, contours, index, color, thickness)
cv2.imshow("Contours",img2)

cv2.waitKey(0)
cv2.destroyAllWindows()


############ 03_07 ############

#area, perimeter and centroid of objects for classification
img = cv2.imread('detect_blob.png',1)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
cv2.imshow("Binary", thresh)

_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img2 = img.copy()
index = -1
thickness = 4
color = (255, 0, 255)

objects = np.zeros([img.shape[0], img.shape[1],3], 'uint8')
for c in contours:
	cv2.drawContours(objects, [c], -1, color, -1)

	area = cv2.contourArea(c)
     #True -> closed loop as closed contour
	perimeter = cv2.arcLength(c, True)
    
	M = cv2.moments(c)
     #draw contour middle points for centroid
	cx = int( M['m10']/M['m00'])
	cy = int( M['m01']/M['m00'])
	cv2.circle(objects, (cx,cy), 4, (0,0,255), -1)

	print("Area: {}, perimeter: {}".format(area,perimeter))

cv2.imshow("Contours",objects)

cv2.waitKey(0)
cv2.destroyAllWindows()


############ 03_08 ############

#canny edge -> edge detection algorithm to create separation of objects in image
#algo determined by speed of color changes
#canny creates single pixel wide line at key high gradient areas in image
img = cv2.imread("tomatoes.jpg",1)

#tomatoes separated by running threshold using hue channel having hue of 25 or less
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
res,thresh = cv2.threshold(hsv[:,:,0], 25, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Thresh",thresh)
#but the above combined all the tomatoes as one object
#overlap of threshold means contour sees as one object

#lower and upper threshold values of edges taken
edges = cv2.Canny(img, 100, 70)
cv2.imshow("Canny",edges)

cv2.waitKey(0)
cv2.destroyAllWindows()


############ 03_10 ############

#segmentation and object detection
#depends which method to use -> if lighting invariant, scale or rotation invariant, consisted of multiple examples, over filter or under filter objects

img = cv2.imread("fuzzy.png",1)
cv2.imshow("Original",img)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3),0)

thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 205, 1)
cv2.imshow("Binary",thresh)

_, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))

#to draw only large contours thus using filtering
filtered = []
for c in contours:
	if cv2.contourArea(c) < 1000:continue
	filtered.append(c)

print(len(filtered))

objects = np.zeros([img.shape[0],img.shape[1],3], 'uint8')
for c in filtered:
	col = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    #random used to use random colors for 3 channels
	cv2.drawContours(objects,[c], -1, col, -1)
	area = cv2.contourArea(c)
	p = cv2.arcLength(c,True)
	print(area,p)

cv2.imshow("Contours",objects)
	

cv2.waitKey(0)
cv2.destroyAllWindows()




############ 04_03 ############

#Feature recognition and Face detection not recognition
#feartures like lighting and scaling, math of pixels

#feature detection --> template matching -> similar pattern between 2 images
#tm -> taking reference source pic and sliding on the input image vertically, horizontally and taking difference and if sum is zero it shows bright spot

#0 means gray scale
template = cv2.imread('template.jpg',0)
frame = cv2.imread("players.jpg",0)

cv2.imshow("Frame",frame)
cv2.imshow("Template",template)

#template matching command
result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)

#draw the matching location circle using maximum brightness of result image
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
print(max_val,max_loc)
cv2.circle(result,max_loc, 15,255,2)
cv2.imshow("Matching",result)

cv2.waitKey(0)
cv2.destroyAllWindows()


############ 04_05 ############

#Face detection --> Haar cascading -> future based machine learning
#training a classifier with images(training data) with and without faces, learns and extracts features from all images
#cascading to know the descriptive features for face detect of eyes or mouth or distance between face elements

img = cv2.imread("faces.jpeg",1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
path = "haarcascade_frontalface_default.xml"

#loads and initializes cascades of functions and classifiers
face_cascade = cv2.CascadeClassifier(path)

#scaleFactor determine false positive or false negative in face detection
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.10, minNeighbors=5, minSize=(40,40))
print(len(faces))

for (x, y, w, h) in faces:
	cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


############ 04_06 ############

img = cv2.imread("faces.jpeg",1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
path = "haarcascade_eye.xml"

eye_cascade = cv2.CascadeClassifier(path)

#detecting eyes
eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.02,minNeighbors=20,minSize=(10,10))
print(len(eyes))

#circling the eyes 
for (x, y, w, h) in eyes:
	xc = (x + x+w)/2
	yc = (y + y+h)/2
	radius = w/2
	cv2.circle(img, (int(xc),int(yc)), int(radius), (255,0,0), 2)
cv2.imshow("Eyes",img)
cv2.waitKey(0)
cv2.destroyAllWindows()




"""
Applications of OpenCV:
a) machine learning --> supervised ml
   confusion matrix -> true and false positives and negatives
b) text recognition OCR
    -> detect text, segment, wrap, feature extraction, character recognition
c) object tracking and optical flow and scene reconstruction
   for real time applications
d) Reading QR codes
    -> detect QR, segment, wrap, binarize, read QR code data
    

Docs referred:
    docs.opencv.org
    
"""




