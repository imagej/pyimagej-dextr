import numpy as np
import scyjava as sj
import xarray as xr
import cv2

def create_bounding_box(image, points=None, pad=0, zero_pad=False):
    if points is not None:
        inds = np.flip(points.transpose(), axis=0)
    else:
        inds = np.where(image > 0)

    if inds[0].shape[0] == 0:
        return None
        
    if zero_pad:
        x_min_bound = -np.inf
        y_min_bound = -np.inf
        x_max_bound = np.inf
        y_max_bound = np.inf
    else:
        x_min_bound = 0
        y_min_bound = 0
        x_max_bound = image.shape[1] - 1
        y_max_bound = image.shape[0] - 1

    x_min = max(inds[1].min() - pad, x_min_bound)
    y_min = max(inds[0].min() - pad, y_min_bound)
    x_max = min(inds[1].max() + pad, x_max_bound)
    y_max = min(inds[0].max() + pad, y_max_bound)

    return x_min, y_min, x_max, y_max

def bounding_box_crop(image, bounding_box, zero_pad=False):
    # get image bounds
    bounds = (0, 0, image.shape[1] - 1, image.shape[0] -1)

    # create valid bounding box (x_min, y_min, x_max, y_max)
    bounding_box_valid = (max(bounding_box[0], bounds[0]),
                          max(bounding_box[1], bounds[1]), 
                          min(bounding_box[2], bounds[2]), 
                          min(bounding_box[3], bounds[3]))

    if zero_pad:
        # initialize crop size with first 2 dimensions
        crop = np.zeros((bounding_box[3] - bounding_box[1] + 1, bounding_box[2] - bounding_box[0] + 1), dtype=image.dtype)
        offsets = (-bounding_box[0], -bounding_box[1])
    else:
        assert(bounding_box == bounding_box_valid)
        crop = np.zeros((bounding_box_valid[3] - bounding_box_valid[1] + 1, bounding_box_valid[2] - bounding_box_valid[0] + 1), dtype=image.dtype)
        offsets = (-bounding_box_valid[0], -bounding_box_valid[1])

    # not sure what this bit does:
    inds = tuple(map(sum, zip(bounding_box_valid, offsets + offsets)))

    image = np.squeeze(image)
    if image.ndim == 2:
        crop[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1] = image[bounding_box_valid[1]:bounding_box_valid[3] + 1, bounding_box_valid[0]:bounding_box_valid[2] + 1]
    else:
        crop = np.tile(crop[:, :, np.newaxis], [1, 1, 3]) # add 3 rgb channels
        crop[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1, :] = image[bounding_box_valid[1]:bounding_box_valid[3] + 1, bounding_box_valid[0]:bounding_box_valid[2] + 1, :]

    return crop

def resize_image(sample, resolution, flagval=None):
    # resize the incoming numpy array
    if flagval is None:
        if ((sample == 0) | (sample == 1)).all():
            flagval = cv2.INTER_NEAREST
        else:
            flagval = cv2.INTER_CUBIC

    if isinstance(resolution, int):
        tmp = [resolution, resolution]
        tmp[np.argmax(sample.shape[:2])] = int(round(float(resolution)/np.min(sample.shape[:2])*np.max(sample.shape[:2])))
        resolution = tuple(tmp)

    if sample.ndim == 2 or (sample.ndim == 3 and sample.shape[2] == 3):
        sample = cv2.resize(sample, resolution[::-1], interpolation=flagval)
    else:
        tmp = sample
        sample = np.zeros(np.append(resolution, tmp.shape[2]), dtype=np.float32)
        for ii in range(sample.shape[2]):
            sample[:, :, ii] = cv2.resize(tmp[:, :, ii], resolution[::-1], interpolation=flagval)
    
    return sample

def mask_crop(image, mask, relax=0, zero_pad=False):
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, dsize=tuple(reversed(image.shape[:2])), interpolation=cv2.INTER_NEAREST)

    assert(mask.shape[:2] == image.shape[:2])
    bbox = create_bounding_box(mask, pad=relax, zero_pad=zero_pad)

    if bbox is None:
        return None

    crop = bounding_box_crop(image, bbox, zero_pad)
    
    return crop

def image_to_numpy(image, ij_instance):
    """
    Take arrays and return numpys
    """
    if isinstance(image, (np.ndarray, np.generic)):
        return image
    elif isinstance(image, xr.DataArray):
        return image.data
    elif sj.jclass('net.imagej.DefaultDataset').isInstance(image):
        image_xr = ij_instance.py.from_java(image)
        return image_xr.data
    elif sj.jclass('net.imagej.Dataset').isInstance(image):
        image_xr = ij_instance.py.from_java(image)
        return image_xr.data
    elif sj.jclass('ij.ImagePlus').isInstance(image):
        ConvertService = ij_instance.get('org.scijava.convert.ConvertService')
        Dataset = sj.jimport('net.imagej.Dataset')
        ds = ConvertService.convert(image, Dataset)
        ds_xr = ij_instance.py.from_java(ds)
        return ds_xr.data

def create_gaussian(size, sigma=10, center=None, d_type=np.float64):
    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)
    y = y[:, np.newaxis]

    if center is None:
        x0 = y0 = size[0] // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2).astype(d_type)

def create_ground_truth(image, labels, sigma=10, one_mask_per_point=False):
    h, w = image.shape[:2]
    if labels is None:
        ground_truth = create_gaussian((h, w), center=(h // 2, w // 2), sigma=sigma)
    else:
        labels = np.array(labels)
        if labels.ndim == 1:
            labels = labels[np.newaxis]
        if one_mask_per_point:
            ground_truth = np.zeros(shape=(h, w, labels.shape[0]))
            for i in range(labels.shape[0]):
                ground_truth[:, :, i] = create_gaussian((h, w), center=labels[i, :], sigma=sigma)
        else:
            ground_truth = np.zeros(shape=(h, w), dtype=np.float64)
            for i in range(labels.shape[0]):
                ground_truth = np.maximum(ground_truth, create_gaussian((h, w), center=labels[i, :], sigma=sigma))

    ground_truth = ground_truth.astype(dtype=image.dtype)

    return ground_truth

def normalize_image(image, max_value):
    
    normalized_image = max_value * (image -image.min()) / max((image.max() - image.min()), 1e-8)
    return normalized_image