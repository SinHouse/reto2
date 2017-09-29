import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa
from scipy import misc, ndimage
from scipy.misc import imsave
import os

source_dataset_folder = '../source_images'
categories = ['LOGO', ]
destination_folder = '../build'
image_size = (224, 224)


def process_categories():
    """Process source categories."""
    for category in categories:
        category_path = os.path.join(source_dataset_folder, category)
        process_category(category_path)


def process_category(category_path: str):
    """Process root source folder."""
    for subdir, dirs, files in os.walk(category_path):
        if len(files) > 0:
            image_path_list = []
            for file in files:
                image_path = os.path.join(subdir, file)
                image_path_list.append(image_path)
            process_folder(subdir, image_path_list)


def load_image_from_disk(image_path: str) -> np.array:
    """Load image from file to numpy array."""
    img = ndimage.imread(image_path, mode="RGB")
    img = img[0:643, 0:643]
    if image_size is not None:
        img = misc.imresize(img, image_size)
    return img


def write_image_to_disk(image_array: np.array, image_path: str):
    """Writes image to disk."""
    print(image_path)
    print(image_array)
    imsave(image_path, image_array)


def process_folder(folder_path: str, image_path_list: [str]):
    """Process folder image."""
    image_aug = process_images(image_path_list)
    destination_path = folder_path.replace(source_dataset_folder,
                                           destination_folder)
    os.makedirs(destination_path)
    print('full images dimension', image_aug)

    image_counter = 0

    for image_index in range(image_aug.shape[0]):
        image_path = os.path.join(destination_path, 'img_' + str(
            image_counter) + '.png')
        write_image_to_disk(image_aug[image_index], image_path=image_path)
        image_counter += 1


# def process_images():
def process_images(image_path_list):
    """Process current folder."""
    ia.seed(1)

    # Example batch of images.
    # The array has shape (n_files, 64, 64, 3) and dtype uint8.
    images = np.array(
        # [ia.quokka(size=(64, 64)) for _ in range(32)],
        [load_image_from_disk(image_path) for image_path in image_path_list],
        dtype=np.uint8
    )
    print('original images dimensions', images.shape)
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Crop(percent=(0, 0.1)),  # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
                      iaa.GaussianBlur(sigma=(0, 0.5))
                      ),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255),
                                  per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True)  # apply augmenters in random order

    images_aug = seq.augment_images(images)
    print('augmented images dimensions', images_aug.shape)
    return images_aug
