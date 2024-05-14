import os
import cv2
import numpy as np

def load_images_from_folder(folder):
    images = []
    for root, _, filenames in sorted(os.walk(folder)):
        for filename in sorted(filenames):
            # id = int(filename[1:8])
            # batch = 2
            # if id<batch*18 or id>=(batch+1)*18:
            #     continue
            img = cv2.imread(os.path.join(root, filename))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
    return np.array(images)

def save_images_to_folder(images, folder):
    for i, img in enumerate(images):
        if not os.path.exists(folder):
            # If it does not exist, create it
            os.makedirs(folder)
        cv2.imwrite(os.path.join(folder, 'image_{:03d}.png'.format(i)), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))