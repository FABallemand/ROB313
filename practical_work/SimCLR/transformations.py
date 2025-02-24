import random

import torch
import torchvision.transforms as transforms

class RandomTransformation(object):

    def __init__(self, output_size):
        """
        Apply random transformation to an image.
        Inspired by: https://arxiv.org/abs/2002.05709

        Parameters
        ----------
        output_size : int | tuple[int, int]
            Size of the output image
        """
        # Output size
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.bk_size = (int(0.1 * self.output_size[0]),
                        int(0.1 * self.output_size[1]))

        self.resized_crop = transforms.RandomResizedCrop(self.output_size,
                                                scale=(0.08, 1.0))
        self.h_flip = transforms.RandomHorizontalFlip(p=0.5)
        self.v_flip = transforms.RandomVerticalFlip(p=0.5)
        self.blur = transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=self.bk_size, sigma=(0.1, 2.0))],
            p=0.5)

    def get_color_distortion(self, s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
        return color_distort

    def __call__(self, img):
        # Random crop and resize
        img = self.resized_crop(img)

        # Random flip
        img = self.h_flip(img)
        img = self.v_flip(img)

        # Color distortion
        img = self.get_color_distortion(random.random())(img)
            
        # Gaussian blur
        img = self.blur(img)
            
        return img
