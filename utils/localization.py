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

def classify_background(img):
    img = cv2.resize(img, (600, 400)).copy()

    mean_color = img.mean(axis=(0,1))
    gr_red_ratio = mean_color[1]/mean_color[0]
    if gr_red_ratio>1:
         return 1#"Textured"
    elif gr_red_ratio<0.94:
        return 2#"Hand"
    else:
        return 0#"Plain"
    
def localize_coins(images):
    tmp_images = []
    types_list = []
    circles_list = []

    for i, image in enumerate(images):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hue, saturation, value = cv2.split(hsv_image)

        type = classify_background(image)
        types_list.append(type)

        if type == 0 or type == "Plain":
            # Pre-process with Gaussian Blur
            tmp = cv2.GaussianBlur(saturation, (9, 9), 2)

            circles_tmp = cv2.HoughCircles(tmp,cv2.HOUGH_GRADIENT_ALT,1.5,20, param1=100,param2=0.78,minRadius=15,maxRadius=50)

        elif type == 1 or type == "Textured":
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            tmp = (saturation.astype(np.float32)-hue.astype(np.float32)).clip(0,255).astype(np.uint8)
            tmp = clahe.apply(tmp)

            # Pre-process with Gaussian Blur
            tmp = cv2.GaussianBlur(tmp, (3, 3), 2)

            circles_tmp = cv2.HoughCircles(tmp,cv2.HOUGH_GRADIENT_ALT,1.5,20, param1=100,param2=0.8,minRadius=15,maxRadius=50)

        else: # Hand background        
            # Threshold the HSV image to get only skin colors
            skin_mask = ((image[:,:,0].astype(np.float32)-image[:,:,1].astype(np.float32))>20).astype(np.uint8)
            
            # Apply morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            skin_mask = cv2.dilate(skin_mask, kernel, iterations = 5)
            skin_mask = cv2.erode(skin_mask, kernel, iterations = 5)     
            
            # Bitwise-AND mask and original image
            hue = cv2.bitwise_and(hue, hue, mask=skin_mask)

            mask2 = (hue>3) & (hue<110)
            tmp = cv2.bitwise_and(hue, hue, mask=mask2.astype(np.uint8))

            # Apply CLAHE on masked saturation channel
            clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(50, 50))
            tmp= clahe.apply(tmp)

            # Pre-process with Gaussian Blur
            tmp = cv2.GaussianBlur(tmp, (3, 3), 2)

            circles_tmp = cv2.HoughCircles(tmp,cv2.HOUGH_GRADIENT_ALT,1.5,20, param1=400,param2=0.7,minRadius=15,maxRadius=50)

            
        tmp_images.append(tmp)
        if circles_tmp is not None:
                circles = remove_overlapping_circles(circles_tmp[0], epsilon=10)
        else:
            circles = np.array([])
        circles_list.append(circles)

    return circles_list, types_list, tmp_images

def extract_coins(images):
    resized_images = [cv2.resize(img, (600,400)) for img in images]
    coins_coord,_,_ = localize_coins(resized_images)
    coins_coord = [coin_coord.astype(np.int32)*10 for coin_coord in coins_coord]
    coins = []
    masks = []

    N,M = images[0].shape[:2]

    for i in range(len(images)):
        coins_i = []
        mask_i = []
        img = images[i]
        centers = coins_coord[i]
        for x_,y_,r in centers:
            x_ = int(x_)
            y_ = int(y_)
            x = min(M-400,max(400, x_))
            y = min(N-400,max(400, y_))
            coin = img[y-400:y+400, x-400:x+400]
            coin_mask = np.zeros((800,800))
            cv2.circle(coin_mask, (400+x-x_,400+y-y_), int(1.1*r), 1, -1)
                
            coins_i.append(coin)
            mask_i.append(coin_mask)
        coins.append(coins_i)
        masks.append(mask_i)
    return coins,masks,coins_coord