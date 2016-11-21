import scipy.misc, skimage
import numpy as np
import glob, os
import json
from ImageAugmenter import ImageAugmenter
import copy
import sklearn

from utils import load_emoji_data

def duplicate_with_noise(emojis, repeat=1000):
    width, height = emojis["images"][0].shape
    augmenter = ImageAugmenter(width, height, # width and height of the image (must be the same for all images in the batch)
                           hflip=False,    # flip horizontally with 50% probability
                           vflip=False,    # flip vertically with 50% probability
                           scale_to_percent=(0.9, 1.1), # scale the image to 70%-130% of its original size
                           scale_axis_equally=False, # allow the axis to be scaled unequally (e.g. x more than y)
                           rotation_deg=5,    # rotate between -25 and +25 degrees
                           shear_deg=5,       # shear between -10 and +10 degrees
                           translation_x_px=2, # translate between -5 and +5 px on the x-axis
                           translation_y_px=2  # translate between -5 and +5 px on the y-axis
                           )
    ret = copy.deepcopy(emojis)
    for i in range(repeat):
        print "Iter {0}".format(i)
        augmented_emojis = augmenter.augment_batch(np.array(emojis["images"], dtype=np.uint8))
        ret["images"] += list(augmented_emojis)
        ret["labels"] += emojis["labels"]
    images, labels = sklearn.utils.shuffle(ret["images"], ret["labels"])
    ret["images"] = images
    ret["labels"] = labels
    images = []
    for image in ret["images"]:
        images.append(skimage.transform.resize(image, (28, 28), preserve_range=True))
    ret["images"] = images
    return ret

def save_noissy_data(emojis, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i, emoji in enumerate(emojis["images"]):
        scipy.misc.imsave(os.path.join(out_dir, str(i) + ".png"), emoji, format="png")
    with open(os.path.join(out_dir, "labels.json"), "w+") as labels_file:
        json.dump(emojis["labels"], labels_file)

def main():
    data_dir = "data"
    out_dir = "noissy_data"
    emojis = load_emoji_data(data_dir)
    emojis["images"] = emojis["images"][:10]
    emojis["labels"] = emojis["labels"][:10]
    emojis = duplicate_with_noise(emojis)
    save_noissy_data(emojis, out_dir)

if __name__ == "__main__":
    main()