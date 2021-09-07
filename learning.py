# filePath = 'messi5.jpg'
'''Mouse event'''
# import cv2 as cv
# import numpy as np

# def ROI(event,x,y,flags,param):
#     global x1,x2,y1,y2,s,img_roi
#     if event == cv.EVENT_LBUTTONDOWN:
#         x1,y1 = x,y

#     elif event == cv.EVENT_LBUTTONUP:
#             print(x1,y1,x,y)
#             img = cv.imread(filePath)
#             img_roi = img[y1:y,x1:x]
#             cv.namedWindow('roi',cv.WINDOW_FREERATIO)
#             cv.imshow('roi',img_roi)
            
#             x1,y1=0
            
            
            

# img = cv.imread(filePath)
# namewin = 'img'
# cv.namedWindow(namewin)
# cv.setMouseCallback(namewin,ROI)
# while True:
#     cv.imshow(namewin,img)
#     if cv.waitKey(20) & 0xff == ord('q'):
#         break
#     if cv.waitKey(20) & 0xff == ord('r'):
#         roi_img = img[0:100,0:400]
#         cv.imshow('tpo',roi_img)
# cv.destroyAllWindows
'''track bar'''
# import cv2 as cv
# import numpy as np

# def notthing(x):
#     pass
# img = np.zeros((512,512,3), np.uint8)
# cv.namedWindow('img')
# cv.createTrackbar('R','img',0,255,notthing)
# cv.createTrackbar('G','img',0,255,notthing)
# cv.createTrackbar('B','img',0,255,notthing)
# swiths ='0: OFF\n1: ON'
# cv.createTrackbar(swiths,'img',0,1,notthing)

# while(1):
#     cv.imshow('img',img)
#     r = cv.getTrackbarPos('R','img')
#     g = cv.getTrackbarPos('G','img')
#     b = cv.getTrackbarPos('B','img')
#     s = cv.getTrackbarPos(swiths,'img')

#     k = cv.waitKey(1) & 0xff 
#     if k == 27:
#         break
#     if s == 0:
#         img[:] = 0
#     else:
#         img[:] = [r,g,b]
#     if k == ord('p'):
#         px = img[100,100]
#         print(px)
# cv.destroyAllWindows()

'''modifying pixel value'''
# import cv2 as cv
# import numpy as np

# img = cv.imread('messi5.jpg')
# # px = img[100,100]
# # print(px)
# img[i,i] =[255,255,255]
# cv.imshow('img',img)
# cv.waitKey()
# cv.destroyAllWindows()
'''Image Properties'''
# import cv2 as cv
# import numpy as np

# img = cv.imread('messi5.jpg')
# print(img.shape)
# print(img.size)
# print(img.dtype)

'''Image ROI'''
# import cv2 as cv
# import numpy as np

# img = cv.imread(filePath)
# # ball = img [280:340,330:390]
# # ball = [255,255,255]
# # img [280:340,330:390] = [255,255,255]
# cv.imshow('img',img)
# cv.waitKey()
# cv.destroyAllWindows()

'''Image Blending'''
# import cv2 as cv
# import numpy as np

# img1 =cv.imread('ml.png')
# img2 = cv.imread('opencv-logo.png')
# dst = cv.addWeighted(img1,0.7,img2,0.3,0)
# cv.imshow('img',dst)
# cv.imshow('img',img2)
# cv.waitKey(0)
# cv.destroyAllWindows()
'''Bitwise'''
# import cv2 
# import numpy as np

# img1 = cv.imread('messi5.jpg')
# img2 = cv.imread('opencv-logo.png')

# rows,cols,channels = img2.shape
# roi = img1[0:rows,0:cols]

# img2gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
# ret,mask = cv.threshold(img2gray,10,255,cv.THRESH_BINARY)
# mask_inv = cv.bitwise_not(mask)

# img1_bg = cv.bitwise_and(roi,roi,mask = mask_inv)

# img2_fg = cv.bitwise_and(img2,img2,mask = mask)
# dst = cv.add(img1_bg,img2_fg)
# img1[0:rows,0:cols] = dst

# cv.imshow('rst',img1)
# cv.waitKey()
# cv.destroyAllWindows()

# rectangle = np.zeros((300, 300), dtype="uint8")
# cv2.rectangle(rectangle, (25, 25), (275, 275), 255, -1)
# cv2.imshow("Rectangle", rectangle)
# # draw a circle
# circle = np.zeros((300, 300), dtype = "uint8")
# cv2.circle(circle, (150, 150), 150, 255, -1)
# cv2.imshow("Circle", circle)
# bitwiseAnd = cv2.bitwise_or(circle,rectangle)
# cv2.imshow("AND", bitwiseAnd)
# cv2.waitKey(0)

# e1 = cv2.getTickCount()
# # your code execution
# e2 = cv2.getTickCount()
# time = (e2 - e1)/ cv2.getTickFrequency()
# img1 = cv2.imread('messi5.jpg')
# e1 = cv2.getTickCount()
# for i in range(5,49,2):
#     img1 = cv2.medianBlur(img1,i)
#     e2 = cv2.getTickCount()
#     t = (e2 - e1)/cv2.getTickFrequency()
#     print(t)
'''Color Object tracking'''
# import cv2 as cv
# import numpy as np
# def nothing(x):
#     pass
# cv.namedWindow('Bluelow')
# cv.namedWindow('Blueup')
# cv.createTrackbar('r_l','Bluelow',0,255,nothing)
# cv.createTrackbar('g_l','Bluelow',0,255,nothing)
# cv.createTrackbar('b_l','Bluelow',0,255,nothing)

# cv.createTrackbar('r_u','Blueup',0,255,nothing)
# cv.createTrackbar('g_u','Blueup',0,255,nothing)
# cv.createTrackbar('b_u','Blueup',0,255,nothing)
# cap = cv.VideoCapture(0)
# while True:
#     ret,frame = cap.read()
#     gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
#     hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
#     r_l = cv.getTrackbarPos('r_l','Bluelow')
#     g_l = cv.getTrackbarPos('g_l','Bluelow')
#     b_l = cv.getTrackbarPos('b_l','Bluelow')

#     r_u = cv.getTrackbarPos('r_u','Blueup')
#     g_u = cv.getTrackbarPos('g_u','Blueup')
#     b_u = cv.getTrackbarPos('b_u','Blueup')
#     low_bl = np.array([60,100,100])
#     up_bl = np.array([70,255,255])
#     mask1 = cv.inRange(hsv,low_bl,up_bl)
#     res = cv.bitwise_and(frame,frame,mask = mask1)

#     cv.imshow('camera',frame)
#     cv.imshow('mask',mask1)
#     cv.imshow('res',res)
#     cv.imshow('gray',hsv)
#     if cv.waitKey(1) & 0xff == ord('q'):
#         break
# cv.destroyAllWindows
'''find value HSV'''
# import cv2 as cv
# import numpy as np
# green = np.uint8([[[0,0,255]]])
# hsv_green = cv.cvtColor(green,cv.COLOR_BGR2HSV)
# print(hsv_green)
'''Threshold'''
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('messi5.jpg',0)
ret,thresh1 = cv.threshold(img,125,255,cv.THRESH_BINARY)
ret,thresh2 = cv.threshold(img,125,255,cv.THRESH_BINARY_INV)
ret,thresh3 = cv.threshold(img,125,255,cv.THRESH_TRUNC)
ret,thresh4 = cv.threshold(img,125,255,cv.THRESH_TOZERO)
ret,thresh5 = cv.threshold(img,125,255,cv.THRESH_TOZERO_INV)

titles = ['img','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
image = [img,thresh1,thresh2,thresh3,thresh4,thresh5]

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(image[i],'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()
