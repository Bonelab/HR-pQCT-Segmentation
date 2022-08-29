import numpy as np


class SamplePadder(object):
    # this transformation object will pad the image and the masks so that:
    # - they are the same size
    # - they are properly aligned
    # - they are square
    # - the side length is a multiple of <img_size_mult>
    # optionally, the transformation can augment the image sample by randomly
    # padding more or less on either side of the x and y dimensions, this is
    # a good idea so that the model is not biased based on how well-centered
    # the training samples are
    def __init__(self, img_size_mult, pad_mode='edge', augmentation=False):
        self.img_size_mult = img_size_mult
        self.pad_mode = pad_mode
        self.augmentation = augmentation

    def __call__(self, sample):
        # when a sample is passed through this transformation, the output
        # should be of images and masks that are all of the same size, aligned,
        # and who have x and y dim lengths the same and a multiple of the integer
        # provided at object initialization.

        volumes = ['image', 'cort_mask', 'trab_mask']
        volumes = list(set(volumes).intersection(sample.keys()))

        # Step 1: Figure out the upper and lower x,y bounds for the union of
        # the image and masks

        lower_x = min([sample[f"{v}_position"][0] for v in volumes])
        upper_x = max([sample[f"{v}_position"][0] + sample[v].shape[0] for v in volumes])

        lower_y = min([sample[f"{v}_position"][1] for v in volumes])
        upper_y = max([sample[f"{v}_position"][1] + sample[v].shape[1] for v in volumes])

        # Step 2: Calculate the new size for the image, which will be the larger
        # of the width and height, then rounded up to the next highest multiple

        width = upper_x - lower_x
        height = upper_y - lower_y

        padded_size = self.img_size_mult * math.ceil(max(width, height) / self.img_size_mult)

        # Step 3: Calculate the new upper/lower bounds in x and y

        lower_x_padded = lower_x - (padded_size - width) // 2
        upper_x_padded = lower_x_padded + padded_size

        lower_y_padded = lower_y - (padded_size - height) // 2
        upper_y_padded = lower_y_padded + padded_size

        # Step 4: Pad the image and masks to the new bounds, and adjust the
        # position entries in the sample dict

        for v in volumes:

            if v == 'image':
                pad_mode = self.pad_mode
            else:
                pad_mode = 'constant'

            sample[v] = np.pad(
                sample[v],
                (
                    (
                        sample[f"{v}_position"][0] - lower_x_padded,
                        upper_x_padded - (sample[f"{v}_position"][0] + sample[v].shape[0])
                    ),
                    (
                        sample[f"{v}_position"][1] - lower_y_padded,
                        upper_y_padded - (sample[f"{v}_position"][1] + sample[v].shape[1])
                    ),
                    (0, 0)
                ),
                mode=pad_mode
            )

            sample[f'{v}_position'][0] = lower_x_padded
            sample[f'{v}_position'][1] = lower_y_padded

        # Step 5: If augmentation is enabled, then randomly pad the image
        # to the next size multiple, to de-center the actual bone

        if self.augmentation:

            # we will randomly choose whether to pad the left/right top/bottom
            # in x and y, using numpy's random binomial function to flip a coin
            # independently for each

            if np.random.binomial(1, 0.5):
                aug_pad_x_lower = self.img_size_mult
                aug_pad_x_upper = 0
            else:
                aug_pad_x_lower = 0
                aug_pad_x_upper = self.img_size_mult

            if np.random.binomial(1, 0.5):
                aug_pad_y_lower = self.img_size_mult
                aug_pad_y_upper = 0
            else:
                aug_pad_y_lower = 0
                aug_pad_y_upper = self.img_size_mult

            # then we loop through the volumes and apply the augmentation pad

            for v in volumes:

                if v == 'image':
                    pad_mode = self.pad_mode
                else:
                    pad_mode = 'constant'

                sample[v] = np.pad(
                    sample[v],
                    (
                        (aug_pad_x_lower, aug_pad_x_upper),
                        (aug_pad_y_lower, aug_pad_y_upper),
                        (0, 0)
                    ),
                    mode=pad_mode
                )

                sample[f'{v}_position'][0] = sample[f'{v}_position'][0] - aug_pad_x_lower
                sample[f'{v}_position'][1] = sample[f'{v}_position'][1] - aug_pad_y_lower

        return sample
