import os
import numpy as np
import scyjava as sj
import xarray as xr
import cv2

class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'pascal':
            return '/path/to/PASCAL/VOC2012'
        elif database == 'sbd':
            return '/path/to/SBD/'
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError


    @staticmethod
    def models_dir():
        return 'models/'


def java_to_numpy(java_image, ij_instance):
    """
    Convert ImageJ java images to numpy arrays uint8 grayscale images.
    """
    # ImageJ resources
    Dataset = sj.jimport('net.imagej.Dataset')
    ImagePlus = sj.jimport('ij.ImagePlus')

    if isinstance(java_image, Dataset):
        xarr_image = ij_instance.py.from_java(java_image)
        if len(xarr_image.shape) <= 2:
            rgb_image = xarr_image.data[:, :, None] * np.ones(3, dtype=np.uint8)[None, None, :]
            return np.array(rgb_image / np.amax(rgb_image) * 255, np.uint8)# scale image to [0,255]
        else:
            return xarr_image.data
    elif isinstance(java_image, ImagePlus):
        ds_image = ij_instance.convert().convert(java_image, Dataset)
        xarr_image = ij_instance.py.from_java(ds_image)
        if len(xarr_image.shape) <= 2:
            rgb_image = xarr_image.data[:, :, None] * np.ones(3, dtype=np.uint8)[None, None, :]
            return np.array(rgb_image / np.amax(rgb_image) * 255, np.uint8)# scale image to [0,255]
        else:
            return xarr_image.data
        

def create_bounding_box(image, points: np.ndarray, pad: int, zero_pad: bool):
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


def crop_bounding_box(image: np.ndarray, bounding_box, zero_pad: bool):
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


def resize_image(image: np.ndarray, resolution: tuple, flagval=None):
    # resize the incoming numpy array
    if isinstance(image, xr.DataArray):
        image = image.data

    if flagval is None:
        if ((image == 0) | (image == 1)).all():
            flagval = cv2.INTER_NEAREST
        else:
            flagval = cv2.INTER_CUBIC

    # convert an int into a tuple --> (x, x)
    if isinstance(resolution, int):
        tmp = [resolution, resolution]
        tmp[np.argmax(image.shape[:2])] = int(round(float(resolution)/np.min(image.shape[:2])*np.max(image.shape[:2])))
        resolution = tuple(tmp)

    if image.ndim == 2:
        image = cv2.resize(image, resolution[::-1], interpolation=flagval)
        image = image[:, :, np.newaxis]
    elif image.ndim == 3 and image.shape[2] ==3:
        image = cv2.resize(image, resolution[::-1], interpolation=flagval)
    else:
        tmp = image
        image = np.zeros(np.append(resolution, tmp.shape[2]), dtype=np.float32)
        for ii in range(image.shape[2]):
            image[:, :, ii] = cv2.resize(tmp[:, :, ii], resolution[::-1], interpolation=flagval)
    
    return image


def crop_from_mask(image, mask, relax=0, zero_pad=False):
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, dsize=tuple(reversed(image.shape[:2])), interpolation=cv2.INTER_NEAREST)

    assert(mask.shape[:2] == image.shape[:2])
    bbox = create_bounding_box(mask, pad=relax, zero_pad=zero_pad)

    if bbox is None:
        return None

    crop = crop_bounding_box(image, bbox, zero_pad)
    
    return crop


def crop_to_binary(image, bbox, im=None, im_size=None, zero_pad=False, relax=0, mask_relax=True, interpolation=cv2.INTER_CUBIC, scikit=False):
    if scikit:
        from skimage.transform import resize as sk_resize
    assert(not(im is None and im_size is None)), 'You have to provide an image or the image size'
    if im is None:
        im_si = im_size
    else:
        im_si = im.shape
    # Borers of image
    bounds = (0, 0, im_si[1] - 1, im_si[0] - 1)

    # Valid bounding box locations as (x_min, y_min, x_max, y_max)
    bbox_valid = (max(bbox[0], bounds[0]),
                  max(bbox[1], bounds[1]),
                  min(bbox[2], bounds[2]),
                  min(bbox[3], bounds[3]))

    # Bounding box of initial mask
    bbox_init = (bbox[0] + relax,
                 bbox[1] + relax,
                 bbox[2] - relax,
                 bbox[3] - relax)

    if zero_pad:
        # Offsets for x and y
        offsets = (-bbox[0], -bbox[1])
    else:
        assert((bbox == bbox_valid).all())
        offsets = (-bbox_valid[0], -bbox_valid[1])

    # Simple per element addition in the tuple
    inds = tuple(map(sum, zip(bbox_valid, offsets + offsets)))

    if scikit:
        image = sk_resize(image, (bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1), order=0, mode='constant').astype(image.dtype)
    else:
        image = cv2.resize(image, (bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1), interpolation=interpolation)
    result_ = np.zeros(im_si)
    result_[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1] = \
        image[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1]

    result = np.zeros(im_si)
    if mask_relax:
        result[bbox_init[1]:bbox_init[3]+1, bbox_init[0]:bbox_init[2]+1] = \
            result_[bbox_init[1]:bbox_init[3]+1, bbox_init[0]:bbox_init[2]+1]
    else:
        result = result_

    return result


def image_to_numpy(image, ij_instance):
    """
    Take arrays and return numpys
    """
    if isinstance(image, (np.ndarray, np.generic)):
        return image
    elif isinstance(image, xr.DataArray):
        return image.data
    elif sj.jclass('net.imagej.DefaultDataset').isInstance(image):
        image = ij_instance.py.from_java(image)
        return image.data
    elif sj.jclass('net.imagej.Dataset').isInstance(image):
        image = ij_instance.py.from_java(image)
        return image.data
    elif sj.jclass('ij.ImagePlus').isInstance(image):
        image = ij_instance.py.to_dataset(image)
        image = ij_instance.py.from_java(image)
        return image.data


def overlay_masks(im, masks, alpha=0.5):
    colors = np.load(os.path.join(os.path.dirname(__file__), 'pascal_map.npy'))/255.
    
    if isinstance(masks, np.ndarray):
        masks = [masks]

    assert len(colors) >= len(masks), 'Not enough colors'

    ov = im.copy()
    im = im.astype(np.float32)
    total_ma = np.zeros([im.shape[0], im.shape[1]])
    i = 1
    for ma in masks:
        ma = ma.astype(np.bool)
        fg = im * alpha+np.ones(im.shape) * (1 - alpha) * colors[i, :3]   # np.array([0,0,255])/255.0
        i = i + 1
        ov[ma == 1] = fg[ma == 1]
        total_ma += ma

        # [-2:] is s trick to be compatible both with opencv 2 and 3
        contours = cv2.findContours(ma.copy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        cv2.drawContours(ov, contours[0], -1, (0.0, 0.0, 0.0), 1)
    ov[total_ma == 0] = im[total_ma == 0]

    return ov


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