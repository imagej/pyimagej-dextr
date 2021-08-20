"""
TorchIJ provides a set of wrapper functions to facilitate in the integration of
PyTorch and ImageJ via PyImageJ.
"""

import numpy as np
import scyjava as sj
import torchij.util
import xarray as xr
from matplotlib import pyplot as plt

def results_to_dataset(results, ij_instance, show=False):
    """
    Send result mask from DEXTR to PyImageJ.
    :param result mask: result_mask from DEXTR
    :param ij_instance: Instance of ImageJ via PyImageJ
    :param show: Show result mask in ImageJ
    :return: net.imagej.DefaultDataset
    """
    results_ds = [] # return a list of Datasets if given list of numpy arrays
    if type(results) == list:
        for result in results:
            result_int = result.astype(int)
            result_ds = ij_instance.py.to_dataset(result_int)
            results_ds.append(results_ds)
            if show:
                ij_instance.ui().show('result mask', result_ds)
        
        return results_ds

    result_int = results.astype(int)
    result_ds = ij_instance.py.to_dataset(result_int)
    if show:
        ij_instance.ui().show('result mask', result_ds)
        
    return result_ds

def preds_to_dataset(predictions, ij_instance, show=False):
    """
    Send prediction from DEXTR to PyImagej.
    :param prediction: Predicition from DEXTR
    :param ij_instance: Instance of ImageJ via PyImageJ
    :param show: Show prediction in ImageJ
    :return: net.imagej.DefaultDataset
    """
    predictions_ds = [] # return a list of Datasets if given list of numpy arrays
    if type(predictions) == list:
        for prediction in predictions:
            prediction_ds = ij_instance.py.to_dataset(prediction)
            predictions_ds.append(prediction_ds)
            if show:
                ij_instance.ui().show('prediction', prediction_ds)
                ij_instance.py.run_macro("""run("Fire");""")

        return predictions_ds

    prediction_ds = ij_instance.py.to_dataset(predictions)
    if show:
        ij_instance.ui().show('prediction', prediction_ds)
        ij_instance.py.run_macro("""run("Fire");""")

    return prediction_ds

def create_extreme_point_window(image, ij_instance, title=None, axis='off'):
    """
    Generate window to collect extreme points around the object
    of interest.
    :param dataset: ImageJ dataset
    :param ij_instance: Instance of ImageJ via PyImageJ
    :param title: Title of image
    :param axis: Display axis on/off
    """
    # check image type
    if isinstance(image, (np.ndarray, np.generic)):
        display_image = image
    elif isinstance(image, xr.DataArray):
        display_image = image.data
    elif sj.jclass('net.imagej.DefaultDataset').isInstance(image):
        if title == None:
            title = str(image.getName())
        image_xr = ij_instance.py.from_java(image)
        display_image = image_xr.data
    elif sj.jclass('net.imagej.Dataset').isInstance(image):
        if title == None:
            title = str(image.getName())
        image_xr = ij_instance.py.from_java(image)
        display_image = image_xr.data
    elif sj.jclass('ij.ImagePlus').isInstance(image):
        if title == None:
            title = str(image.getTitle())
        convertService = ij_instance.get('org.scijava.convert.ConvertService')
        Dataset = sj.jimport('net.imagej.Dataset')
        ds = convertService.convert(image, Dataset)
        ds_xr = ij_instance.py.from_java(ds)
        display_image = ds_xr.data

    # display image and collect points
    plt.ion()
    plt.axis(axis)
    plt.imshow(display_image)
    plt.title(title)

    return

def collect_extreme_points_ori(points=4, timeout=0):
    """
    Collect the extreme points from an open
    pyplot window and return origins.
    :param points: Number of points to collect (int)
    :param timeout: Set timeout (int)
    """
    extreme_points_ori = np.array(plt.ginput(points,timeout=timeout)).astype(np.int)

    return extreme_points_ori

def mask_to_dataset(mask, ij_instance, show=False):
    """
    Return mask as a dataset.
    :param mask: Boolean numpy array
    """
    if mask.dtype == bool:
        mask = mask.astype(int)
        
    mask_ds = ij_instance.py.to_dataset(mask)
    if show:
        ij_instance.ui().show(mask_ds)
    
    return mask_ds

def mask_to_xarray(mask, ij_instance, show=False):
    """
    Return mask as xarray DataArray.
    """
    if mask.dtype == bool:
        mask = mask.astype(int)

    mask_ds = ij_instance.py.to_dataset(mask)
    if show:
        ij_instance.ui().show(mask_ds)

    mask_xr = ij_instance.py.from_java(mask_ds)

    return mask_xr

def join_mask_to_image(image, mask, ij_instance, show=False):
    """
    Take the result mask or masks from the neural network and return an imagej stack.
    Returns joined Dataset.
    """
    if isinstance(mask, np.ndarray):
        mask = mask_to_dataset(mask, ij_instance)
        mask = ij_instance.py.from_java(mask)

        if len(mask.dims) <= 2:
            mask = mask.expand_dims('Channel')

        joined_image = xr.concat([image[:,:,:], mask[:,:,:]], dim='Channel')
        joined_image = ij_instance.py.to_dataset(joined_image)

        if show:
            ij_instance.ui().show(joined_image)
    elif isinstance(mask, list):
        if len(mask) <= 1:
            single_mask = mask[0]
            single_mask = mask_to_xarray(single_mask, ij_instance)
            
            if len(single_mask.dims) <= 2:
                single_mask = single_mask.expand_dims('Channel')

            joined_image = xr.concat([image[:,:,:], single_mask[:,:,:,]], dim='Channel')
            joined_image = ij_instance.py.to_dataset(joined_image)
        elif len(mask) >=2:
            mask_list_ds = []
            for m in mask: # convert numpy list to net.imagej.Dataset list
                m = mask_to_xarray(m, ij_instance)
                if len(m.dims) <= 2:
                    m = m.expand_dims('Channel')
                mask_list_ds.append(m)
            
            mask_count = len(mask_list_ds)
            temp_mask = mask_list_ds[0]
            for index in range(1, mask_count): # Dorky way to "append" masks to the input image
                temp_mask = xr.concat([temp_mask[:,:,:], mask_list_ds[index][:,:,:]], dim='Channel')

            joined_image = xr.concat([image[:,:,:], temp_mask[:,:,:]], dim='Channel')
            joined_image = ij_instance.py.to_dataset(joined_image)
        
        if show:
            ij_instance.ui().show(joined_image)

    return joined_image