import imagej
import scyjava as sj
import numpy as np
import net_segment as ns
import torch
import matplotlib
from matplotlib import pyplot as plt
from torch.nn.functional import interpolate

matplotlib.interactive(True)

# define some network parameters
model_name = "dextr_pascal-sbd"
pad = 35
thres = 0.3
gpu_id = 0
device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")

# create the network and load the weights (demo)
net = ns.construct_resnet101(model_name)
net.to(device)

# start imagej
print("Starting ImageJ ...")
ij = imagej.init(mode='interactive')
print(f"ImageJ version: {ij.getVersion()}")
ij.ui().showUI()

# get ImageJ resources
Overlay = sj.jimport('ij.gui.Overlay')

print("Opening image ...")
dataset = ij.io().open('doc/sample-data/cell.jpg')
numpy_image = ns.util.java_to_numpy(dataset, ij)
ov = Overlay()
rm = ij.RoiManager.getRoiManager()
imp = ij.py.to_imageplus(dataset)
imp.getProcessor().resetMinAndMax()
imp.show()
ns.create_extreme_point_window(dataset, ij)

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
        result = ns.detection_to_binary(image=numpy_image, prediction=pred, points=extreme_points, zero_pad=True, pad=pad, thres=thres)

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
