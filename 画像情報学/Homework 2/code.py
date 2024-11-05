import cv2
import numpy as np

def transform_to_monochrome_image(input_image_path, output_image_path):

    img = cv2.imread(input_image_path)

    ycbcr_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y, Cb, Cr = cv2.split(ycbcr_img)
    Cb[:] = 128
    Cr[:] = 128

    monochrome_ycbcr_img = cv2.merge([Y, Cb, Cr])
    monochrome_bgr_img = cv2.cvtColor(monochrome_ycbcr_img, cv2.COLOR_YCrCb2BGR)

    cv2.imwrite(output_image_path, monochrome_bgr_img)
    print(f"output the image: {output_image_path}")


input_image_path = 'input_image.png'
output_image_path = 'output_image.png'
transform_to_monochrome_image(input_image_path, output_image_path)
