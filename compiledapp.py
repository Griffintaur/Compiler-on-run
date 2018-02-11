#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np

def ImagePreProcessing(ipath):
#    image=cv2.imread(path)
#    gray_iamge=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#    cv2.adaptiveThreshold(gray_image,255,cv2.THRESH_OTSU,cv2.THRESH_BINARY,10)
#    cv2.imshow(gray_image)
    largeImage = cv2.imread(ipath)
    rgb = cv2.pyrDown(largeImage)
    small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
    
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    # using RETR_EXTERNAL instead of RETR_CCOMP
    contours,hierarchy,_ = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    mask = np.zeros(bw.shape, dtype=np.uint8)
    print(str(len(contours)))
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y+h, x:x+w] = 0
        shape=contours.shape
        contours1 = np.array(contours).reshape((shape[0],shape[1])).astype(np.int32)
        cv2.drawContours(mask, contours1, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
    
        if r > 0.45 and w > 8 and h > 8:
            cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2) 
    cv2.imshow('rects', rgb)
    cv2.waitKey(0)

def ImageTemplateMatching():
    pass

def MainExecution():
    pass

if __name__=='__main__':
    #inputpath=path("/home/singhankit/Desktop/image.jpg")
    print ("START---------------")
    ImagePreProcessing("/home/singhankit/Desktop/image.jpg")