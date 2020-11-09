##################################################################################
import os, shutil, time, sys
import numpy as np
import matplotlib.pyplot as plt

import torchvision, torch
import torch.nn as nn
from skimage.color import lab2rgb
from torchvision import transforms

from helper import LoadImageFromFolder, ScalePixel, AverageMeter, show_img, to_rgb
from model import ColorizerNeuralNet, RegressorNetwork
from prepareData import PrepareData

# Check if GPU is available
use_gpu = torch.cuda.is_available()


# use_gpu = False

####################################################################################


class Main():

    def __init__(self):
        pass

    @staticmethod
    def train(train_loader, model, criterion, optimizer, epoch):
        print('Starting training epoch {}'.format(epoch))
        model.train()

        # Prepare value counters and timers
        batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

        end = time.time()
        for i, (input_gray, input_ab, target, input_ab_mean) in enumerate(train_loader):

            # Use GPU if available
            if use_gpu:
                input_gray, input_ab, target, input_ab_mean = input_gray.cuda(), input_ab.cuda(), target.cuda(), input_ab_mean.cuda()

            # Record time to load data (above)
            data_time.update(time.time() - end)

            # Run forward pass
            output_ab = model(input_gray)
            loss = criterion(output_ab, input_ab)
            losses.update(loss.item(), input_gray.size(0))

            # Compute gradient and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record time to do forward and backward passes
            batch_time.update(time.time() - end)
            end = time.time()

            # Print model accuracy -- in the code below, val refers to value, not validation
            if i % 5 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))
        print('Finished training epoch {}'.format(epoch))

    @staticmethod
    def validate(val_loader, model, criterion, save_images, epoch, user_input=False):
        model.eval()

        # Prepare value counters and timers
        batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

        end = time.time()
        already_saved_images = False
        for i, (input_gray, input_ab, target, input_ab_mean) in enumerate(val_loader):
            data_time.update(time.time() - end)

            # Use GPU
            if use_gpu: input_gray, input_ab, target, input_ab_mean = input_gray.cuda(), input_ab.cuda(), target.cuda(), input_ab_mean.cuda()

            # Run model and record loss
            output_ab = model(input_gray)  # throw away class predictions
            loss = criterion(output_ab, input_ab)
            losses.update(loss.item(), input_gray.size(0))

            # Save val images to file
            if save_images and not already_saved_images:
                already_saved_images = True
                for j in range(min(len(output_ab), 10)):  # save at most 10 images
                    save_path = {'grayscale': '../blue_cis6930/rishab.lokray/outputs/gray/',
                                 'colorized': '../blue_cis6930/rishab.lokray/outputs/color/'}
                    save_name = 'img-{}-epoch-{}.jpg'.format(i * val_loader.batch_size + j, epoch)
                    to_rgb(input_gray[j].cpu(), ab_input=output_ab[j].detach().cpu(), save_path=save_path,
                           save_name=save_name)

            # Record time to do forward passes and save images
            batch_time.update(time.time() - end)
            end = time.time()

            # Print model accuracy -- in the code below, val refers to both value and validation

            print('Validate: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses))

        print('Finished validation.')
        return losses.avg

    @staticmethod
    def main(cnn_type, load_pretrained_model=None):

        # Choosing colorizer using user input
        model = ColorizerNeuralNet()

        # Use GPU if available
        if use_gpu:
            model.cuda()
            print('Loaded model onto GPU.')

        # Create loss function, optimizer #criterion = nn.CrossEntropyLoss().cuda() if use_gpu else nn.CrossEntropyLoss()
        criterion = nn.MSELoss().cuda() if use_gpu else nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)

        # Create transformations to be applied for training
        train_transforms = transforms.Compose([
            transforms.Scale(128)
        ])

        transforms_for_augmentation = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(128),
            ScalePixel(0.6, 1.0)
        ])

        # Loading the original dataset with no transformation
        augmentedSet = LoadImageFromFolder('../blue_cis6930/rishab.lokray/images/train/', train_transforms)

        # Augmenting to the original dataset with transformation if GPU avaiable.
        if use_gpu:
            for i in range(10):
                augmentedSet = torch.utils.data.ConcatDataset(
                    [augmentedSet,
                     LoadImageFromFolder('../blue_cis6930/rishab.lokray/images/train/', transforms_for_augmentation)])

        train_loader = torch.utils.data.DataLoader(augmentedSet, batch_size=64, shuffle=True)
        print('Loaded training data.')

        # Create transformations to be applied for validation
        val_transforms = transforms.Compose([
            transforms.Resize((128, 128))
        ])

        val_imagefolder = LoadImageFromFolder("../blue_cis6930/rishab.lokray/images/val/", val_transforms)
        val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=64, shuffle=False)
        print('Loaded validation data.')

        # Move model and loss function to GPU
        if use_gpu:
            criterion = criterion.cuda()
            model = model.cuda()

        # Make folders and set parameters
        os.makedirs('../blue_cis6930/rishab.lokray/outputs/color', exist_ok=True)
        os.makedirs('../blue_cis6930/rishab.lokray/outputs/gray', exist_ok=True)
        save_images = True
        epochs = 100

        # Load pretrained model
        if load_pretrained_model:
            model_checkpoint = load_pretrained_model
            model.load_state_dict(torch.load("../blue_cis6930/rishab.lokray/checkpoints/{}".format(model_checkpoint)))
            print("LOADED MODEL FROM MEMORY")
            with torch.no_grad():
                validate(val_loader, model, criterion, save_images, 94)
            print("Tested model on testing set, check output/ folder for the original and predicted outputs. \n")

            # Testing on user images.
            user_input = True
            user_imagefolder = LoadImageFromFolder("../blue_cis6930/rishab.lokray/images/userInput/", val_transforms)
            userimage_loader = torch.utils.data.DataLoader(user_imagefolder, batch_size=64, shuffle=False)
            validate(userimage_loader, model, criterion, save_images, 94, cmd_ip)
            print("Tested on user input")
            exit()

        best_losses = float("inf")

        # Train model if no pretrained model available.
        for epoch in range(epochs):
            # Train for one epoch, then validate
            Main.train(train_loader, model, criterion, optimizer, epoch)
            with torch.no_grad():
                losses = Main.validate(val_loader, model, criterion, save_images, epoch)
            # Save checkpoint and replace old best model if current model is better
            if losses < best_losses:
                best_losses = losses
                strr = "colorizerModel"
                torch.save(model.state_dict(),
                           '../blue_cis6930/rishab.lokray/checkpoints/{}-epoch-{}-losses-{:.3f}.pth'.format(strr,
                                                                                                            epoch + 1,
                                                                                                            losses))


# Program starts here.
if __name__ == '__main__':
    PrepareData.test_train_split()
    Main.main("coloriser")

