filePath = 'messi5.jpg'
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