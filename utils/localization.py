import numpy as np
import cv2
import skimage

def check_overlap(circle1, circle2, epsilon=1e-6):
    """
    Check if two circles overlap.
    """
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return distance + epsilon <= r1 + r2

def remove_overlapping_circles(circles, epsilon=1e-6):
    """
    Remove circles that overlap with each other.
    """
    # Create a boolean mask to mark non-overlapping circles
    mask = np.ones(len(circles), dtype=bool)

    # Iterate over each circle
    for i in range(len(circles)):
        if mask[i]:
            for j in range(i + 1, len(circles)):
                if check_overlap(circles[i], circles[j], epsilon=epsilon):
                    # Mark overlapping circle for removal
                    mask[j] = False

    # Return non-overlapping circles
    return circles[mask]

def localize_coins(img):
    N,M = img.shape[:2]
    ratio = 10
    img = cv2.resize(img, (int(M/ratio), int(N/ratio))).copy()

    mean_color = img.mean(axis=(0,1))
    gr_red_ratio = mean_color[1]/mean_color[0]

    red = img[:,:,0].astype(np.float32)
    green = img[:,:,1].astype(np.float32)
    blue = img[:,:,2].astype(np.float32)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue = hsv_img[:,:,0].astype(np.float32)
    saturation = hsv_img[:,:,1].astype(np.float32)
    saturation_th, _= cv2.threshold(saturation.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    value = hsv_img[:,:,2].astype(np.float32)

    if gr_red_ratio>1:
        img_type = 1

        tmp = (saturation-hue).clip(0,255).astype(np.uint8)

        circles_tmp = cv2.HoughCircles(tmp,cv2.HOUGH_GRADIENT_ALT,1.5,20, param1=1,param2=0.49,minRadius=5,maxRadius=50)

    elif gr_red_ratio<0.94:
        img_type = 2      
        
        hand = (red-green)>20 #((hue>130)|(hue<10)).astype(np.uint8)*255 #(hue<10) | (hue>30)
        hand = skimage.morphology.closing(hand, footprint=skimage.morphology.disk(radius=15))
        hand = skimage.morphology.remove_small_holes(hand,area_threshold=20000)

        tmp = hand & (hue>10) & (hue<130)
        tmp = skimage.morphology.remove_small_holes(tmp,area_threshold=1000)
        tmp = skimage.morphology.erosion(tmp, footprint=skimage.morphology.disk(radius=3))
        tmp = skimage.morphology.opening(tmp, footprint=skimage.morphology.disk(radius=3))      

        tmp = tmp.astype(np.uint8)     

        circles_tmp = cv2.HoughCircles(tmp,cv2.HOUGH_GRADIENT_ALT,1.5,20, param1=1,param2=0.58,minRadius=15,maxRadius=50)
    else:
        img_type = 0
        tmp = saturation.astype(np.uint8)
        circles_tmp = cv2.HoughCircles(tmp,cv2.HOUGH_GRADIENT_ALT,1.5,20, param1=1,param2=0.4,minRadius=5,maxRadius=50)

    if circles_tmp is not None:
        circles = remove_overlapping_circles(circles_tmp[0], epsilon=0)
    else:
        circles = []

    return circles*ratio#, img_type