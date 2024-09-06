import numpy as np
import cv2

def gaussian(x, sigma):
    return np.exp(-x*2 / (2 * sigma*2)) / (np.sqrt(2 * np.pi) * sigma)

def fast_joint_bilateral_filter(img_1, img_2, spatial_sigma, range_sigma,d):
    shadow_threshold=50

    height, width, channels = img_1.shape
    output_img = np.zeros_like(img_1, dtype='float64')
    F_base = np.zeros_like(img_1, dtype='float64')
    F_detail = np.zeros_like(img_1, dtype='float64')
    spatial_filter = gaussian(np.arange(-d, d), spatial_sigma)

    for c in range(channels):  
        for j in range(width):  
            filtered_row_1 = np.zeros(height)
            filtered_row_2 = np.zeros(height)

            for i in range(height):
                start = max(0, i - d)
                end = min(height, i + d)
                
                weights = spatial_filter[:end-start]
                normalized_weights = weights / np.sum(weights)

                filtered_row_1[i] = np.sum(normalized_weights * img_1[start:end, j, c])
                filtered_row_2[i] = np.sum(normalized_weights * img_2[start:end, j, c])

            filtered_img_1 = np.zeros(height)
            filtered_img_2 = np.zeros(height)

            for i in range(height):
                
                start = max(0, i - d)
                end = min(height, i + d)
                
                weights = spatial_filter[:end - start]
                normalized_weights = weights / np.sum(weights)

                filtered_img_1[i] = np.sum(normalized_weights * filtered_row_1[start:end])
                filtered_img_2[i] = np.sum(normalized_weights * filtered_row_2[start:end])

            img=np.zeros((height,width,3))
            for i in range(height):
                if(img_1[i,j,c]-img_2[i,j,c]<=shadow_threshold):
                    img[i,j,c]=img_1[i,j,c]
                else:
                    img[i,j,c]=img_2[i,j,c]
            for i in range(height):
                constant_pixel_value_2 = img_2[i, j, c]
                neighboring_pixel_values_2 = img_2[max(0, i - d):min(height, i + d + 1), j, c] 
                intensity_difference_2 = np.abs(constant_pixel_value_2 - neighboring_pixel_values_2)
                range_weights_2 = gaussian(intensity_difference_2, range_sigma)
                F_base[i, j, c] = np.sum(range_weights_2 * filtered_img_2[i]) / np.sum(range_weights_2)

            epsilon=0.02

            for i in range(height):
                F_detail[i,j,c]=(img_2[i,j,c]+epsilon)/(F_base[i,j,c]+epsilon)

            for i in range(height):

                constant_pixel_value = img[i, j, c]
                neighboring_pixel_values = img[max(0, i - d):min(height, i + d), j, c]
                intensity_difference = np.abs(constant_pixel_value - neighboring_pixel_values)
                range_weights = gaussian(intensity_difference, range_sigma)
                output_img[i, j, c] = np.sum(range_weights * filtered_img_1[i]) / np.sum(range_weights)

    
    return np.multiply(output_img,F_detail)


def solution(image_path_a, image_path_b):
    ############################
    ############################
    ## image_path_a is path to the non-flash high ISO image
    ## image_path_b is path to the flash low ISO image
    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    
    
    image_1 = cv2.imread(image_path_a)
    image_2 = cv2.imread(image_path_b)
    image_1 = np.array(image_1)
    image_2 = np.array(image_2)

    image_1 = image_1.astype(np.float64)
    image_2 = image_2.astype(np.float64)

    jbf_img = fast_joint_bilateral_filter(image_1, image_2, spatial_sigma=40, range_sigma=0.25, d=30)

    jbf_img_uint8 = cv2.convertScaleAbs(jbf_img)

    return jbf_img_uint8
