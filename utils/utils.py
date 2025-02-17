import os
import cv2
import numpy as np
import random
import nbformat
from nbconvert import PythonExporter
from IPython import get_ipython

def load_images_from_folder(folder, max_images=-1, batch_idx=0, resize=True):
    images = []
    cnt = 0

    for root, _, filenames in sorted(os.walk(folder)):
        for filename in sorted(filenames):
            if cnt < batch_idx:
                cnt += 1
                continue
            if max_images == -1 or len(images) < max_images:
                img = cv2.imread(os.path.join(root, filename))
                if img is not None:
                    if resize:
                        img = cv2.resize(img, (600, 400))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
            else:
                break
    return np.array(images)

def save_images_to_folder(images, folder):
    for i, img in enumerate(images):
        if not os.path.exists(folder):
            # If it does not exist, create it
            os.makedirs(folder)

        if len(np.shape(img)) < 3 or np.shape(img)[2] == 1:
            cv2.imwrite(os.path.join(folder, 'image_{:03d}.png'.format(i)), img)
        else:
            cv2.imwrite(os.path.join(folder, 'image_{:03d}.png'.format(i)), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def select_random_images(images, types=None, target_type=-1, max_count=-1, seed=-1):
    #sanity check
    if types is not None and len(images) != len(types) and target_type != -1:
        raise ValueError('The number of images and types should be the same')

    images = images.copy()
    
    if types is None:
        types = [target_type for _ in range(len(images))]
    else:
        types = types.copy()
    
    if seed != -1:
        random.seed(seed)
        shuffled_indices = random.sample(range(len(images)), k=len(images))
        images = [images[i] for i in shuffled_indices]
        types = [types[i] for i in shuffled_indices]
        random.seed() # Reset the seed

    selected_images = []
    for image, type_ in zip(images, types):
        if type_ == target_type or target_type == -1:
            selected_images.append(image)
            if len(selected_images) == max_count and max_count != -1:
                break
    return selected_images

def train_test_split(images,labels,masks=None,seed=-1,ratio=0.8):
    if seed != -1:
        random.seed(seed)
        
    shuffled_indices = random.sample(range(len(images)), k=len(images))
    random.seed() # Reset the seed

    images = [images[i] for i in shuffled_indices]        
    labels = [labels[i] for i in shuffled_indices]

    train_images = images[:int(len(images)*ratio)]
    test_images = images[int(len(images)*ratio):]
    train_labels = labels[:int(len(labels)*ratio)]
    test_labels = labels[int(len(labels)*ratio):]

    if masks is not None:
        masks = [masks[i] for i in shuffled_indices]
        train_masks = masks[:int(len(masks)*ratio)]
        test_masks = masks[int(len(masks)*ratio):]
        
        return train_images, test_images, train_labels, test_labels, train_masks, test_masks
    
    return train_images, test_images, train_labels, test_labels

def run_notebook(path):
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)
    
    exporter = PythonExporter()
    code, _ = exporter.from_notebook_node(nb)
    
    ip = get_ipython()
    ip.run_cell(code)
