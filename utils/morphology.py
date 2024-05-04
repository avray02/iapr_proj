import skimage

def remove_objects(image,radius=1000):

    image = skimage.morphology.opening(image, skimage.morphology.disk(radius=radius))
    image = skimage.morphology.closing(image, skimage.morphology.disk(radius=radius))

    return image


def keep_objects(image,min_area=1000,max_area=8000):

    image = skimage.morphology.remove_small_objects(image, min_size=min_area)
    removed_objects = skimage.morphology.remove_small_objects(image, min_size=max_area)

    return image-removed_objects
