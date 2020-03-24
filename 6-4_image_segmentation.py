import cv2
import numpy as np
from matplotlib import pyplot as plt

filename = 'rice.png'
image = cv2.imread(filename)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#大津算法灰度阈值化
thr, bw = cv2.threshold(gray, 0, 0xff, cv2.THRESH_OTSU)
print('Threshold is:', thr)

#画出灰度图
plt.hist(gray.ravel(), 256, [0, 256])
plt.show()

element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, element)

# seg = copy.deepcopy(bw)
seg = bw
#计算轮廓
cnts, hier = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

count = 0
#遍历所有区域，并过滤较小区域
for i in range(len(cnts), 0, -1):
    c = cnts[i-1]
    area = cv2.contourArea(c)
    if area < 10:
        continue
    count = count + 1
    print('blob', i, ":", area)

    #区域画框标记
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(image, (x, y), (x+w, y+h), (0,0,0xff), 1)
    cv2.putText(image, str(count), (x,y), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 0xff, 0))

print("米粒数量:", count)
cv2.imshow("原图", image)
cv2.imshow("阈值图", bw)

cv2.waitKey()
cv2.destroyAllWindows()
