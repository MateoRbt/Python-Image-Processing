import cv2
import numpy as np
import PIL
from PIL import ImageOps
from PIL import Image



def solarize(originalImage, thresValue): #Solarization function
    sol = Image.fromarray(originalImage) #convert image from array and put into variable
    solar = PIL.ImageOps.solarize(sol, thresValue )  # solarize the image on variable "sol" with the given threshold
    solar.save('solar'+ (str(thresValue)) + '.jpg', "JPEG")# save the image in the current folder as solar(threshnumber).jpg
    solar.show() #show the image
    return

imageToUseName = 'napoli.jpg'
originalImag = cv2.imread(imageToUseName)
img_GRAY = cv2.cvtColor(originalImag, cv2.COLOR_BGR2GRAY) #convert image to grayscale
filename = 'napoli_gray.jpg'  #declare filename
cv2.imwrite(filename, img_GRAY) # export image in folder

#show the images
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.imshow("Original Image", originalImag)
cv2.resizeWindow("Original Image", 480, 360)
cv2.namedWindow("Grayscale Image", cv2.WINDOW_NORMAL)
cv2.imshow("Grayscale Image", img_GRAY)
cv2.resizeWindow("Grayscale Image", 480, 360)
cv2.waitKey(0)
cv2.destroyAllWindows()


solarize(img_GRAY, 64) #solarize function 64 threshold
solarize(img_GRAY, 128) # solarize function 128 threshold
solarize(img_GRAY, 192) # solarize function 192 threshold

exit() #exit the program
