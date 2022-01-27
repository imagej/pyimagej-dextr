import os
import imagej
import scyjava as sj
import numpy as np
import net_segment as ns
import net_segment.networks.deeplab2 as resnet
import torch
from net_segment.util import Path
from collections import OrderedDict
from matplotlib import pyplot as plt
from torch.nn.functional import interpolate

def show_ij_img(array):
    ds = ij.py.to_dataset(array)
    ij.ui().show(ds)

    return

def show_plt_img(array):
    plt.imshow(array)
    plt.show()

    return

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

# start imagej
print("Starting ImageJ ...")
ij = imagej.init(mode='interactive')
print(f"ImageJ version: {ij.getVersion()}")
ij.ui().showUI()

# get ImageJ resources
ImagePlus = sj.jimport('ij.ImagePlus')
Overlay = sj.jimport('ij.gui.Overlay')
RoiManager = sj.jimport('ij.plugin.frame.RoiManager')()

print("Opening image ...")
ij_image = ij.io().open('sample-data/cell.jpg')
numpy_image = ns.util.java_to_numpy(ij_image, ij)
ov = Overlay()
rm = RoiManager.getRoiManager()
imp = ij.convert().convert(ij_image, ImagePlus)
imp.show()
ns.create_extreme_point_window(ij_image, ij)

results = []

with torch.no_grad():
    while 1:
        print("Collecting points ...")
        extreme_points = ns.collect_extreme_points(points=5)

        # crop image to bounding box from extreme_points
        print("Creating crop ...")
        crop = ns.crop_to_extreme_points(image=numpy_image, points=extreme_points, pad=pad, zero_pad=True)

        # generate extreme point heat map
        print("Generating heatmap ...")
        heatmap = ns.create_extreme_point_heatmap(image=crop, resolution=(512, 512), points=extreme_points, pad=pad)

        # convert inputs to tensor
        print("Converting inputs ...")
        inputs = ns.convert_to_tensor(crop, heatmap, resolution=(512, 512))

        # run a forward pass
        print("Sending inputs to the network ...")
        inputs = inputs.to(device)
        outputs = net.forward(inputs)
        outputs = interpolate(outputs, size=(512, 512), mode='bilinear',align_corners=True)
        outputs = outputs.to(torch.device('cpu'))

        pred = np.transpose(outputs.data.numpy()[0, ...], (1, 2, 0))
        pred = 1 / (1 + np.exp(-pred))
        pred = np.squeeze(pred)
        result = ns.detection_to_binary(image=numpy_image, prediction=pred, points=extreme_points, zero_pad=True, pad=pad)

        # convert masks to roi anJ
        roi = ns.roi.mask_to_roi(result, ij)
        rm.addRoi(roi)
        ov.add(roi)
        imp.setOverlay(ov)

        print("Collecting masks ...")
        results.append(result)
        
        # plot the results
        print("Displaying masks ...")
        plt.imshow(ns.util.overlay_masks(numpy_image / 255, results))
        plt.plot(extreme_points[:, 0], extreme_points[:, 1], 'gx')
