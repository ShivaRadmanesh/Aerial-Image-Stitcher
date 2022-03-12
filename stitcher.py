import argparse
from math import log

import cv2
import imutils
import numpy as np
from imutils import paths


def powerLaw(img):
    def gamma_finder(img):
        mean = cv2.mean(img)[0]   
        gamma = log(mean) / log(127)
        print(gamma)
        return gamma

    def max(img):
        r, c = img.shape
        max = 0

        for i in range(r):
            for j in range(c):
                if (img[i, j] > max):
                    max = img[i, j]
        return max

    gamma = gamma_finder(img)

    r, c = img.shape

    output = np.zeros((r, c))

    for i in range(r):
        for j in range(c):
            output[i, j] =  img[i, j] ** gamma
    
    # convert to output image data type from float64 to uint8
    output_uint8 = cv2.convertScaleAbs(output)

    # normalize th image
    nomalization_coef = 255 / max(output) + 6
    output_uint8 *= round(nomalization_coef)
    return output_uint8
    

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True, help="Path to input directory of images to stitch")
ap.add_argument("-o", "--output", type=str, required=True, help="Path to the output image")
ap.add_argument("-e", "--enhance", 
    help="Enhance the quality of the image using power law and gama methods",
    action="store_true",
    default=False
)
args = vars(ap.parse_args())

# grab the paths to the input images and initialize our images list
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["images"])))
images = []

# loop over the image paths, load each one, and add them to our
# images to stitch list
for imagePath in imagePaths:
    image = cv2.imread(imagePath, 0)
    
    if args['enhance']:
        image = powerLaw(image)

    images.append(image)
        

# initialize OpenCV's image stitcher object and then perform the image
# stitching
print("[INFO] stitching images...")
stitcher = cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)

# if the status is '0', then OpenCV successfully performed image
# stitching
if status == 0:
	# write the output stitched image to disk
	# display the output stitched image to our screen
	cv2.imshow("Stitched", stitched)
	cv2.waitKey(0)

	try:
		cv2.imwrite(args["output"], stitched)
	except:
		pass

# otherwise the stitching failed, likely due to not enough keypoints)
# being detected
else:
	print("[INFO] image stitching failed ({})".format(status))
