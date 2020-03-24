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
img = cv2.imread(filename)
np.save('lena.npy', img)
img = gauss_noise(img)

blur = cv2.blur(img, (5,5))
gauss = cv2.GaussianBlur(img, (5,5), 0)
median = cv2.medianBlur(img, 5)
bilateral = cv2.bilateralFilter(img, 5, 150, 150)

cv2.imshow("Image", img)
cv2.imshow("Blurred", blur)
cv2.imshow("Gauss", gauss)
cv2.imshow("Median filtered", median)
cv2.imshow("Bilateral filtered", bilateral)

cv2.waitKey()
cv2.destroyAllWindows()
