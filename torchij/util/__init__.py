import numpy as np

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

def crop_image(image, bounding_box, zero_pad=False):
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