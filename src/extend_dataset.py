"""Extendiendo dataset."""
import Augmentor

p = Augmentor.Pipeline("../source_images/LOGO")


def extend_set():
    p.rotate(probability=0.6, max_left_rotation=25, max_right_rotation=25)
    p.rotate_random_90(probability=0.15)
    # p.rotate90(probability=0.5)
    # p.rotate270(probability=0.5)
    p.shear(probability=1.0, max_shear_left=25, max_shear_right=25)
    p.flip_left_right(probability=0.2)
    # p.flip_top_bottom(probability=0.1)
    # p.crop_random(probability=1, percentage_area=0.5)
    p.resize(probability=1.0, width=128, height=128)
    p.random_distortion(probability=0.25, grid_height=4, grid_width=4,
                        magnitude=3)

    # Here we sample 100,000 images from the pipeline.

    # It is often useful to use scientific notation for specify
    # large numbers with trailing zeros.
    # num_of_samples = int(1e4)
    num_of_samples = 2000

    # Now we can sample from the pipeline:
    p.sample(num_of_samples)
