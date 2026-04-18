import imageio, json
from PIL import Image
import argparse
import albumentations as A
from helper_functions import read_leads, convert_bounding_boxes_to_dict, rotate_bounding_box, get_lead_pixel_coordinate, rotate_points
import numpy as np
import matplotlib.pyplot as plt
import os, sys, argparse
from scipy.io import savemat, loadmat
from matplotlib.ticker import AutoMinorLocator
from math import ceil
import time
import random


def _kelvin_to_rgb_multipliers(kelvin):
    """Convert a colour temperature in Kelvin to (r, g, b) multipliers in [0, 1].

    Port of the Tanner Helland approximation used by imgaug's
    ChangeColorTemperature. Input clamped to the documented valid range
    [1000, 40000] K.
    """
    kelvin = float(max(1000, min(40000, kelvin))) / 100.0
    if kelvin <= 66:
        r = 255.0
        g = 99.4708025861 * np.log(kelvin) - 161.1195681661
        b = 0.0 if kelvin <= 19 else 138.5177312231 * np.log(kelvin - 10) - 305.0447927307
    else:
        r = 329.698727446 * np.power(kelvin - 60, -0.1332047592)
        g = 288.1221695283 * np.power(kelvin - 60, -0.0755148492)
        b = 255.0
    return (
        float(np.clip(r, 0, 255) / 255.0),
        float(np.clip(g, 0, 255) / 255.0),
        float(np.clip(b, 0, 255) / 255.0),
    )


def _apply_color_temperature(img, kelvin):
    """Multiply RGB channels by Kelvin-derived factors (expects HxWx3 RGB)."""
    r, g, b = _kelvin_to_rgb_multipliers(kelvin)
    out = img.astype(np.float32).copy()
    out[..., 0] *= r
    out[..., 1] *= g
    out[..., 2] *= b
    return np.clip(out, 0, 255).astype(np.uint8)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_directory', type=str, required=True)
    parser.add_argument('-i', '--input_file', type=str, required=True)
    parser.add_argument('-o', '--output_directory', type=str, required=True)
    parser.add_argument('-r','--rotate',type=int,default=25)
    parser.add_argument('-n','--noise',type=int,default=25)
    parser.add_argument('-c','--crop',type=float,default=0.01)
    parser.add_argument('-t','--temperature',type=int,default=6500)
    return parser

# Main function for running augmentations
def get_augment(input_file,output_directory,rotate=25,noise=25,crop=0.01,temperature=6500,bbox=False, store_text_bounding_box=False, json_dict=None):
    filename = input_file
    image = Image.open(filename)
    
    image = np.array(image)
    
    lead_bbs = []
    leadNames_bbs = []
    
         
    lead_bbs, leadNames_bbs, lead_bbs_labels, startTime_bbs, endTime_bbs, plotted_pixels = read_leads(json_dict['leads'])

    rgb_image = image[:, :, :3]
    h, w, _ = image.shape
    rot = random.randint(-rotate, rotate)
    crop_sample = random.uniform(0, crop)
    # Image-only augmentation pipeline (bounding-box transforms are handled
    # manually via rotate_bounding_box below, matching the original logic).
    # albumentations 2.x expresses GaussNoise stddev in normalized [0, 1]
    # whereas imgaug used [0, 255] on uint8. Divide `noise` by 255.
    noise_std = float(noise) / 255.0
    transform = A.Compose([
        A.Affine(rotate=(rot, rot), p=1.0),
        A.GaussNoise(std_range=(noise_std, noise_std), p=1.0),
        A.CropAndPad(percent=-crop_sample, sample_independently=False, p=1.0),
    ])
    augmented = transform(image=rgb_image)["image"]
    augmented = _apply_color_temperature(augmented, temperature)

    if bbox:
        augmented_lead_bbs = rotate_bounding_box(lead_bbs, [h/2,w/2], -rot)
    else:
        augmented_lead_bbs = []    
    if store_text_bounding_box:
        augmented_leadName_bbs = rotate_bounding_box(leadNames_bbs, [h/2,w/2], -rot)
    else:
        augmented_leadName_bbs = []   

    rotated_pixel_coordinates = rotate_points(plotted_pixels, [h/2, w/2], -rot)

    if bbox or store_text_bounding_box:
        json_dict['leads'] = convert_bounding_boxes_to_dict(augmented_lead_bbs, augmented_leadName_bbs, lead_bbs_labels, startTime_bbs, endTime_bbs, rotated_pixel_coordinates)

    head, tail = os.path.split(filename)

    f = os.path.join(output_directory,tail)
    plt.imsave(fname=f,arr=augmented)

    return f

