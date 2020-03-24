import cv2
import numpy as np

img = cv2.imread('source.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thr = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
# cv2.imshow('Source iamge', img)
# cv2.imshow('Threshold image', thr)

# 获得图像轮廓
cnts, hier = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
count = 0
# 显示图像
disp_poly = img.copy()
disp_elli = img.copy()

for i in range(len(cnts)):
    c = cnts[i]
    poly = cv2.approxPolyDP(c, 5, True)
    cv2.polylines(disp_poly, [poly], True, (255,255,255), 2)

    if (len(c)>5):
        ellipse = cv2.fitEllipse(c)
    cv2.ellipse(disp_elli, ellipse, (255,255,255), 2)
# cv2.imshow('Polygon', disp_poly)
# cv2.imshow('Ellipse ', disp_elli)

    # 计算Hu不变距
    area = cv2.contourArea(c)
    length = cv2.arcLength(c, True)
    moments = cv2.moments(c)
    hu = cv2.HuMoments(moments)
    print(i + 1, ":","length=%.1f" %length, "area=", area,
    "m00=%.3f, m01=%.3f, m10=%.3f, mll=%.3f" %(hu[0],hu[1],hu[2],hu[3]))

    # 获取包围框并在左上角显示序号
    x, y, w, h, = cv2.boundingRect(c)
    cv2.putText(disp_poly, str(i+1), (x,y), cv2.FONT_HERSHEY_PLAIN, 0.8, (0xff,0xff,0xff))

cv2.imshow('Polygon fitting result', disp_poly)
cv2.imshow('Ellipse result', disp_elli)
cv2.waitKey()
cv2.destroyAllWindows()
