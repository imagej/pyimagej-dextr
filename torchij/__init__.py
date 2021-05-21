"""
TorchIJ provides a set of wrapper functions to facilitate in the integration of
PyTorch and ImageJ via PyImageJ.
"""

import numpy as np

def result_mask_to_dataset(results, ij_instance, show=False):
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
    result_ds = ij_instance.py.to_dataset(result_ds)
    if show:
        ij_instance.ui().show('result mask', result_ds)
        
    return result_ds

def pred_to_dataset(prediction, ij_instance, show=False):
    """
    Send prediction from DEXTR to PyImagej.
    :param prediction: Predicition from DEXTR
    :param ij_instance: Instance of ImageJ via PyImageJ
    :param show: Show prediction in ImageJ
    :return: net.imagej.DefaultDataset
    """
    prediction_ds = ij_instance.py.to_dataset(prediction)
    if show:
        ij_instance.ui().show('prediction', prediction_ds)
        ij_instance.py.run_macro("""run("Fire");""")
    return prediction_ds