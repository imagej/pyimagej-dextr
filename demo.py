import os
import imagej
import numpy as np
import torchij as tij
import torchij.networks.deeplab2 as resnet
import torch
from mypath import Path
from collections import OrderedDict
from matplotlib import pyplot as plt
from torch.nn.functional import upsample

# start imagej
print("Starting ImageJ ...")
ij = imagej.init(headless=False)
print(f"ImageJ version: {ij.getVersion()}")

def show_ij_img(array):
    ds = ij.py.to_dataset(array)
    ij.ui().show(ds)

    return

def show_plt_img(array):
    plt.imshow(array)
    plt.show()

    return

def load_img(path):
    img_ds = ij.io().open(path)
    img_xr = ij.py.from_java(img_ds)
    print(f"Complete ... \nArray shape: {img_xr.shape}\nArray ndim: {img_xr.ndim}")

    return img_xr
    
# define some network parameters
model_name = "dextr_pascal-sbd"
pad = 50
thres = 0.8
gpu_id = 0
device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")

####################################################
# create the network and load the weights (demo)
net = resnet.resnet101(1, nInputChannels=4, classifier='psp') # constructs a ResNet-101 model
print("Initializing weights from: {}".format(os.path.join(Path.models_dir(), model_name + '.pth')))
state_dict_chkpnt = torch.load(os.path.join(Path.models_dir(), model_name + '.pth'), map_location=lambda storage, loc: storage)

# remove the `module.` prefix from the model -- if trained using DataParallel
if 'module.' in list(state_dict_chkpnt.keys())[0]:
    new_state_dict = OrderedDict()
    for k, v in state_dict_chkpnt.items():
        name = k[7:]  # remove `module.` from multi-gpu training
        new_state_dict[name] = v
else:
    new_state_dict = state_dict_chkpnt
    
net.load_state_dict(new_state_dict)
net.eval()
net.to(device)
####################################################

# open image and collect points
print("Opening image ...")
image = load_img('imgs/cell.jpg')
tij.create_extreme_point_window(image, ij, title="Data")

results = []

with torch.no_grad():
    while 1:
        print("Collecting points ...")
        extreme_points_ori = tij.collect_extreme_points_ori()

        # crop image to bounding box from extreme_points_ori
        print("Creating crop ...")
        bbox = tij.util.create_bounding_box(image, points=extreme_points_ori, pad=pad, zero_pad=True)
        crop = tij.util.crop_bounding_box(image, bounding_box=bbox, zero_pad=True)
        resize = tij.util.resize_image(crop, (512, 512)).astype(np.float32)

        # generate extreme point heat map
        print("Generating heatmap ...")
        extreme_points = extreme_points_ori - [np.min(extreme_points_ori[:, 0]),  np.min(extreme_points_ori[:1])] + [pad, pad]
        extreme_points = (512 * extreme_points * [1 / crop.shape[1], 1 / crop.shape[0]]).astype(int)
        heatmap = tij.util.create_ground_truth(resize, extreme_points, sigma=10)
        heatmap = tij.util.normalize_image(heatmap, 255)

        # convert inputs to tensor
        print("Converting inputs ...")
        dextr_input = np.concatenate((resize, heatmap[:, :, np.newaxis]), axis=2)
        inputs = torch.from_numpy(dextr_input.transpose((2, 0, 1))[np.newaxis, ...])

        # run a forward pass
        print("Sending inputs to the network ...")
        inputs = inputs.to(device)
        outputs = net.forward(inputs)
        outputs = upsample(outputs, size=(512, 512), mode='bilinear', align_corners=True)
        outputs = outputs.to(torch.device('cpu'))

        pred = np.transpose(outputs.data.numpy()[0, ...], (1, 2, 0))
        pred = 1 / (1 + np.exp(-pred))
        pred = np.squeeze(pred)
        result = tij.util.crop_to_mask(pred, bbox, im_size=image.shape[:2], zero_pad=True, relax=pad) > thres

        print("Collecting masks ...")
        results.append(result)

        # plot the results
        print("Displaying masks ...")
        plt.imshow(tij.util.overlay_masks(image.data / 255, results))
        plt.plot(extreme_points_ori[:, 0], extreme_points_ori[:, 1], 'gx')