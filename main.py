import cv2
import numpy as np
import random
import time
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error


# Functions
def sp_noise(image,prob): #salt and paper function
    output = np.zeros(image.shape,np.uint8) #array of zeros
    thres = 1 - prob #threshold val
    for i in range(image.shape[0]): #calculate pixel value for new image
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def speckle_noise(image2): #speckle function
    gauss = np.random.normal(0, 1, image2.size)
    gauss = gauss.reshape(image2.shape[0], image2.shape[1]).astype('uint8')
    noisy = image2 + image2 * gauss
    return noisy


#import image
imageToUseName = "sopranos.jpg"
originalImag = cv2.imread(imageToUseName)

#Convert To grayscale
img_GRAY = cv2.cvtColor(originalImag, cv2.COLOR_BGR2GRAY)
filename = 'SopranosGray.jpg'
cv2.imwrite(filename, img_GRAY)

#Show images

cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.imshow("Original Image", originalImag)
cv2.resizeWindow("Original Image", 480, 360)

cv2.namedWindow("Grayscale Image", cv2.WINDOW_NORMAL)
cv2.imshow("Grayscale Image", img_GRAY)
cv2.resizeWindow("Grayscale Image", 480, 360)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Noise addition
snpImg = sp_noise(img_GRAY, 0.10)
speckleImg = speckle_noise(img_GRAY)

# Show noisy images
cv2.namedWindow("SP noise Image", cv2.WINDOW_NORMAL)
cv2.imshow("SP noise Image", snpImg)
cv2.resizeWindow("SP noise Image", 480, 360)
filename = 'SP.jpg'
cv2.imwrite(filename, snpImg)

cv2.namedWindow("Speckle noise Image", cv2.WINDOW_NORMAL)
cv2.imshow("Speckle noise Image", speckleImg)
cv2.resizeWindow("Speckle noise Image", 480, 360)
filename = 'Speckle.jpg'
cv2.imwrite(filename, speckleImg)

cv2.waitKey(0)
cv2.destroyAllWindows()

#Filtering session

# Avg filter
kernel = np.ones((5,5),np.float32)/25 #averaging kernel for 5 x 5 window patch
t = time.time()
snpavg = cv2.filter2D(snpImg, -1, kernel)
specleavg = cv2.filter2D(speckleImg, -1, kernel)
tmpRunTime = time.time() - t

#Show images after AVG Filtering
cv2.namedWindow("Averaging Filter", cv2.WINDOW_NORMAL)
cv2.imshow("Averaging Filter", snpavg)
cv2.resizeWindow("Averaging Filter", 480, 360)
filename = 'Avg1.jpg'
cv2.imwrite(filename, snpavg)

cv2.namedWindow("Averaging Filter 2", cv2.WINDOW_NORMAL)
cv2.imshow("Averaging Filter 2", specleavg)
cv2.resizeWindow("Averaging Filter 2", 480, 360)
filename = 'Avg2.jpg'
cv2.imwrite(filename, specleavg)

# SSIM AND MSE for s&p
(score, _) = ssim(img_GRAY, snpavg, full=True) #ssim
print( " \n .. Averaging blur filter time was: {:.8f}".format(tmpRunTime), " seconds. SSIM sccre: {:.4f}".format(score))

Y = mean_squared_error(img_GRAY, snpavg) #mse
print("MSE for image set 1 - AVG Blur:", Y, "\n")

# SSIM AND MSE for speckle

(score, _) = ssim(img_GRAY, specleavg, full=True) #ssim
print( " .. Averaging blur filter 2 time was: {:.8f}".format(tmpRunTime), " seconds. SSIM sccre: {:.4f}".format(score))

Y = mean_squared_error(img_GRAY, specleavg) # mse
print("MSE for image set 2 - AVG Blur:", Y, "\n")

cv2.waitKey(0)
cv2.destroyAllWindows()

#GAUSS filter

t = time.time()
snpgauss = cv2.GaussianBlur(snpImg, (5, 5), 0)
speclegauss = cv2.GaussianBlur(speckleImg, (5, 5), 0)
tmpRunTime = time.time() - t
#illustrate the results
cv2.namedWindow("Gauss blur Filter", cv2.WINDOW_NORMAL)
cv2.imshow("Gauss blur Filter", snpgauss)
cv2.resizeWindow("Gauss blur Filter", 480, 360)
filename = 'Gauss1.jpg'
cv2.imwrite(filename, snpgauss)

cv2.namedWindow("Gauss blur Filter 2", cv2.WINDOW_NORMAL)
cv2.imshow("Gauss blur Filter 2", speclegauss)
cv2.resizeWindow("Gauss blur Filter 2", 480, 360)
filename = 'Gauss2.jpg'
cv2.imwrite(filename, speclegauss)

# SSIM AND MSE for s&p
(score, _) = ssim(img_GRAY, snpgauss, full=True) #ssim
print( " \n .. Gauss blur filter time was: {:.8f}".format(tmpRunTime), " seconds. SSIM sccre: {:.4f}".format(score))

Y = mean_squared_error(img_GRAY, snpgauss) #mse
print("MSE for image set 1 - Gauss Blur:", Y, "\n")

# SSIM AND MSE for speckle

(score, _) = ssim(img_GRAY, speclegauss, full=True) #ssim
print( " .. Gauss blur filter 2 time was: {:.8f}".format(tmpRunTime), " seconds. SSIM sccre: {:.4f}".format(score))

Y = mean_squared_error(img_GRAY, speclegauss)# mse
print("MSE for image set 2 - Gauss Blur:", Y, "\n")

cv2.waitKey(0)
cv2.destroyAllWindows()

# Median filter

median = cv2.medianBlur(snpImg, 5)
median2 = cv2.medianBlur(speckleImg,5)

cv2.namedWindow("Median blur Filter", cv2.WINDOW_NORMAL)
cv2.imshow("Median blur Filter", median )
cv2.resizeWindow("Median blur Filter", 480, 360)
filename = 'Med1.jpg'
cv2.imwrite(filename, median)

cv2.namedWindow("Median blur Filter 2", cv2.WINDOW_NORMAL)
cv2.imshow("Median blur Filter 2", median2 )
cv2.resizeWindow("Median blur Filter 2", 480, 360)
filename = 'Med2.jpg'
cv2.imwrite(filename, median2)


# SSIM AND MSE for s&p
(score, _) = ssim(img_GRAY, median, full=True) #ssim
print( " \n .. Median  blur filter time was: {:.8f}".format(tmpRunTime), " seconds. SSIM sccre: {:.4f}".format(score))

Y = mean_squared_error(img_GRAY, median) #mse
print("MSE for image set 1 - Median Blur:", Y, "\n")

# SSIM AND MSE for speckle

(score, _) = ssim(img_GRAY, median2, full=True) #ssim
print( " .. Median blur filter 2 time was: {:.8f}".format(tmpRunTime), " seconds. SSIM sccre: {:.4f}".format(score))

Y = mean_squared_error(img_GRAY, median) # mse
print("MSE for image set 2 - Median Blur:", Y, "\n")

cv2.waitKey(0)
cv2.destroyAllWindows()