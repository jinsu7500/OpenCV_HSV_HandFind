import cv2 as cv
import numpy as np
import os

#살색영역 검출
def make_mask_image(img_bgr):    
  img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
  low = (0, 30, 0)
  high = (15, 255, 255)
  img_mask = cv.inRange(img_hsv, low, high)
  return img_mask

#바이너리에서 검출된 살생역역테두리중 가장 큰영역찾기(손찾기)
def findMaxArea(contours):
  
  max_contour = None
  max_area = -1

  for contour in contours:
    area = cv.contourArea(contour)

    x,y,w,h = cv.boundingRect(contour)

    if (w*h)*0.4 > area:
        continue

    if w > h:
        continue

    if area > max_area:
      max_area = area
      max_contour = contour
  
  if max_area < 10000:
    max_area = -1

  return max_area, max_contour


def process(img_bgr):
  img_result = img_bgr.copy()
  # STEP 2 살색영역 검출
  img_binary = make_mask_image(img_bgr)
  # STEP 3 ( 검출된 살색영역 이진화)
  kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
  img_binary = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel, 1)
  cv.imshow("Binary", img_binary) ##바이너리영상 출력

  # STEP 4 바이너리 이미지에서 흰색영역의 외각선(파랑)표시
  contours, hierarchy = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  for cnt in contours:
    cv.drawContours(img_result, [cnt], 0, (255, 0, 0), 3)   

  # STEP 5 손영역 검출
  max_area, max_contour = findMaxArea(contours)  
  if max_area == -1:
    return img_result

  hull = cv.convexHull(max_contour)
  cv.drawContours(img_result, [hull], 0, (0,255,0), 2)
  return img_result


# STEP 1 카메라로
cap = cv.VideoCapture(0)
while True:

  ret,img_bgr = cap.read()
  
  if ret == False:
    break

  img_result = process(img_bgr)
  
  key = cv.waitKey(1) 
  if key== 27:
      break

  cv.imshow("Result", img_result)

cap.release()
cv.destroyAllWindows()
