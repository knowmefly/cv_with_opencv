import cv2
import numpy as np

def gauss_noise(image, mean=0, var=0.001):
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var**0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)

    return out

filename = 'lena.jpg'
img = cv2.imread(filename, 0)

# #Sobel算子边缘检测
# sobel = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
# #Laplacian边缘检测
# laplacian = cv2.Laplacian(img, cv2.CV_16S)
# #Canny边缘检测，最小阈值50，最大阈值为120
# canny = cv2.Canny(img, 50, 120)
#
# sobel_show = cv2.convertScaleAbs(sobel)
# lap_show = cv2.convertScaleAbs(laplacian)
# cv2.imshow('Sobel', sobel_show)
# cv2.imshow('Laplician', lap_show)
#
# cv2.imshow('Canny', canny)


#形态学滤波
nimg = gauss_noise(img)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
eroded = cv2.erode(img, kernel)
dilated = cv2.dilate(img, kernel)

opened = cv2.morphologyEx(nimg, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

# # 腐蚀、膨胀、 开、闭运算
# cv2.imshow('Noised image', nimg)
# cv2.imshow('Eroded image', eroded)
# cv2.imshow('Dilated image', dilated)
# cv2.imshow('Opened image', opened)
# cv2.imshow('Closed image', closed)

#计算并显示梯度、顶帽和黑帽变换
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
cv2.imshow('Image gradient', gradient)
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
bottomhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
cv2.imshow('Tophat', tophat)
cv2.imshow('Blackhat', bottomhat)
# 顶帽和黑帽变换进行图像增强，并显示结果
enhanced = img + tophat - bottomhat
cv2.imshow('Enhanced image', enhanced)
# 计算击中或击不中（HMT）变换结果并显示
kernel = np.array(([0,1,0], [1,-1,1], [0,1,0]), dtype='int')
hmt = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)
cv2.imshow('Hit or miss transform', hmt)

cv2.waitKey()
cv2.destroyAllWindows()
