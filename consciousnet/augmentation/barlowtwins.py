# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Define a set of transforms.
"""

# Imports
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms


class ContrastiveConnectomeTransform(object):
    """ Tranforms that can be applied on connectome for contrastive tasks.
    """
    def __init__(self, scale=(0.1, 0.3), value=0):
        """ Init class.

        Parameters
        ----------
        scale: 2-uplet
            range of proportion of erased connections against input matrix
            size.
        value: float, default 0
            erasing value.
        """
        self.transform = transforms.Compose([
            RandomConnectionErasing(scale=scale, value=value)])
        self.transform_prime = transforms.Compose([
            RandomConnectionErasing(scale=scale, value=value)])

    def __call__(self, x):
        """ Apply the transforms.

        Parameters
        ----------
        x: array
            the input array.

        Returns
        -------
        y1, y2: arrays
            the transformed inputs.
        """
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


class ContrastiveImageTransform(object):
    """ Tranforms that can be applied on images for contrastive tasks.
    """
    def __init__(self, crop_size):
        """ Init class.

        Parameters
        ----------
        crop_size: int
            the image cropt size.
        """
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(
                crop_size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8),
            transforms.RandomGrayscale(p=0.2),
            RandomGaussianBlur(p=1.0),
            RandomSolarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(
                crop_size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8),
            transforms.RandomGrayscale(p=0.2),
            RandomGaussianBlur(p=0.1),
            RandomSolarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    def __call__(self, x):
        """ Apply the transforms.

        Parameters
        ----------
        x: array
            the input array.

        Returns
        -------
        y1, y2: arrays
            the transformed inputs.
        """
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


class RandomConnectionErasing(object):
    """ Randomly erases connections.
    """
    def __init__(self, scale=(0.1, 0.3), value=0):
        """ Init class.

        Parameters
        ----------
        scale: 2-uplet
            range of proportion of erased connections against input matrix
            size.
        value: float, default 0
            erasing value.
        """
        self.scale = scale
        self.value = value

    def _add_noise(self, matrix):
        ratio = np.random.uniform(low=self.scale[0], high=self.scale[1])
        nb_of_erasing = int(matrix.size * ratio)
        noise_locs = np.random.choice(
            matrix.size, size=nb_of_erasing, replace=False)
        noise = np.ones((matrix.size, ), dtype=matrix.dtype)
        noise[noise_locs] = self.value
        return matrix * noise.reshape(matrix.shape)

    def __call__(self, matrix):
        """ Apply the transform.

        Parameters
        ----------
        matrix: array
            the input connectome.

        Returns
        -------
        transformed: array
            the transformed connectome.
        """
        assert matrix.shape[0] == 1
        m = self._add_noise(matrix)
        return self._add_noise(matrix)


class RandomGaussianBlur(object):
    """ Random Gaussian blur transform.
    """
    def __init__(self, p):
        """ Init class.

        Parameters
        ----------
        p: float
            apply the transform with this probability [0, 1].
        """
        self.p = p

    def __call__(self, img):
        """ Apply the transform.

        Parameters
        ----------
        img: Image
            the input image.

        Returns
        -------
        transformed: Image
            the transformed input image.
        """
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class RandomSolarization(object):
    """ Random solarization transform.
    """
    def __init__(self, p):
        """ Init class.

        Parameters
        ----------
        p: float
            apply the transform with this probability [0, 1].
        """
        self.p = p

    def __call__(self, img):
        """ Apply the transform.

        Parameters
        ----------
        img: Image
            the input image.

        Returns
        -------
        transformed: Image
            the transformed input image.
        """
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
