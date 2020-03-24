import cv2
from matplotlib import pyplot as plt
filename = 'lena.jpg'
img = cv2.imread(filename)
cv2.imshow('Source', img)

# # 实现图像改变大小和翻转
w, h = img.shape[0:2]
# resized = cv2.resize(img, (int(w/4), int(h/2)))
# flipped = cv2.flip(img, -1)
#
# cv2.imshow("Resized", resized)
# cv2.imshow("Flipped", flipped)

# 实现图像的距离变换
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thr = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)
dist = cv2.distanceTransform(thr, cv2.DIST_L2, cv2.DIST_MASK_3)
dist_norm = cv2.convertScaleAbs(dist)

# # 实现Log-polar变换
# center = (w/2, h/2)
# maxRadius = 0.7*min(center)
# M = w/cv2.log(maxRadius)
# print(maxRadius, M[0])
# log_polar = cv2.logPolar(img, center, M[0]*0.8, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
# cv2.imshow('Distance transform', dist_norm)
# cv2.imshow('Log-polar', log_polar)

# # 实现灰度直方图和直方图均衡化
# plt.hist(gray.ravel(), 256, [0,256])
# plt.show()
# equa = cv2.equalizeHist(gray)
# cv2.imshow('Equalized image', equa)

# 实现Hough变换
edges = cv2.Canny(thr, 50, 150)
disp_edge = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
line = cv2.HoughLinesP(edges, 1, 1*np.pi/180, 10)
cv2.waitKey()
cv2.destroyAllWindows()
