import os, time, shutil, argparse
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

from skimage.color import rgb2lab, rgb2gray, lab2rgb

from torchvision import datasets, transforms


class LoadImageFromFolder(datasets.ImageFolder):

    def __getitem__(self, item):
        path, target = self.imgs[item]
        image = self.loader(path)

        if self.transform is not None:
            image_original = self.transform(image)
            image_original = np.asarray(image_original)
            image_lab = rgb2lab(image_original)
            image_lab = (image_lab + 128) / 255
            image_ab = image_lab[:, :, 1:3]
            image_ab = torch.from_numpy(image_ab.transpose((2, 0, 1))).float()
            image_gray = rgb2gray(image_original)
            image_gray = torch.from_numpy(image_gray).unsqueeze(0).float()

        if self.target_transform is not None:
            target = self.target_transform(target)

        #Calculating the mean of a and b tensor
        a_mean = torch.mean(image_ab[0])
        b_mean = torch.mean(image_ab[1])

        return image_gray, image_ab, target, torch.tensor([a_mean, b_mean])


class ScalePixel(object):
    def __init__(self, begin_range, end_range):
        self.begin_range = begin_range
        self.end_range = end_range

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be scaled by a scalar.
        Returns:
            Tensor: Converted image.
        """

        b = torch.tensor(random.uniform(self.begin_range, self.end_range), dtype=torch.float32)
        scaled_tensor = transforms.ToTensor()(pic) * b
        return transforms.ToPILImage()(scaled_tensor)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class AverageMeter(object):
    '''An easy way to compute and store both average and current values'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def visualize_image(grayscale_input, ab_input=None, show_image=False, save_path=None, save_name=None):
#     '''Show or save image given grayscale (and ab color) inputs. Input save_path in the form {'grayscale': '/path/', 'colorized': '/path/'}'''
#     plt.clf()  # clear matplotlib plot
#     ab_input = ab_input.cpu()
#     grayscale_input = grayscale_input.cpu()
#     if ab_input is None:
#         grayscale_input = grayscale_input.squeeze().numpy()
#         if save_path is not None and save_name is not None:
#             plt.imsave(grayscale_input, '{}.{}'.format(save_path['grayscale'], save_name), cmap='gray')
#         if show_image:
#             plt.imshow(grayscale_input, cmap='gray')
#             plt.show()
#     else:
#         color_image = torch.cat((grayscale_input, ab_input), 0).numpy()
#         color_image = color_image.transpose((1, 2, 0))
#         color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
#         color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
#         color_image = lab2rgb(color_image.astype(np.float64))
#         grayscale_input = grayscale_input.squeeze().numpy()
#         if save_path is not None and save_name is not None:
#             plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
#             plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))
#         if show_image:
#             f, axarr = plt.subplots(1, 2)
#             axarr[0].imshow(grayscale_input, cmap='gray')
#             axarr[1].imshow(color_image)
#             plt.show()





def show_img(img):
    plt.figure(figsize=(18, 15))
    # unnormalize
    img = img / 2 + 0.5
    npimg = img.numpy()
    npimg = np.clip(npimg, 0., 1.)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def to_rgb(grayscale_input, ab_input, save_path=None, save_name=None):
    '''Show/save rgb image from grayscale and ab channels
     Input save_path in the form {'grayscale': '/path/', 'colorized': '/path/'}'''
    plt.clf()  # clear matplotlib
    color_image = torch.cat((grayscale_input, ab_input), 0).numpy()  # combine channels
    color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
    color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
    color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
    color_image = lab2rgb(color_image.astype(np.float64))
    grayscale_input = grayscale_input.squeeze().numpy()
    if save_path is not None and save_name is not None:
        plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
        plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))
