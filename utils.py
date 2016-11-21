import glob, os
import scipy.misc

def load_emoji_data(data_dir):
    emojis = {"images": [], "labels": []}
    for image_name in glob.glob(data_dir + "/*.png"):
        emojis["images"].append(scipy.misc.imread(image_name, flatten=True, mode='F'))
        emojis["labels"].append(os.path.splitext(os.path.basename(image_name))[0])
    return emojis
