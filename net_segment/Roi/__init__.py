import numpy as np
import scyjava as sj

def array_to_rectangle_roi(ij_instance, image, rectangle_array: np.ndarray, add_to_roi_manager=False):
    """
    Convert an array of rectangles into ImageJ ROIs.
    """
    # get ImageJ resources
    Dataset = sj.jimport('net.imagej.Dataset')
    ImagePlus = sj.jimport('ij.ImagePlus')
    Roi = sj.jimport('ij.gui.Roi')
    Overlay = sj.jimport('ij.gui.Overlay')
    ov = Overlay()

    if isinstance(image, Dataset):
        imp = ij_instance.convert().convert(image, ImagePlus)
    else:
        imp = image

    # get image dimensions
    dimensions = tuple(imp.getDimensions())
    imp_width = dimensions[0]
    imp_height = dimensions[1]

    if add_to_roi_manager:
        RoiManager = sj.jimport('ij.plugin.frame.RoiManager')()
        rm = RoiManager.getRoiManager()


    for i in range(len(rectangle_array)):
        ymin, xmin, ymax, xmax = rectangle_array[i]
        rect_x = xmin * imp_width
        rect_y = ymin * imp_height
        rect_width = (xmax * imp_height) - (xmin * imp_height)
        rect_height = (ymax * imp_width) - (ymin * imp_width)
        roi = Roi(rect_x, rect_y, rect_width, rect_height)
        imp.setRoi(roi)
        ov.add(roi)
        if add_to_roi_manager:
            rm.addRoi(roi)

    imp.setOverlay(ov)
    imp.show()
    
    return None

def array_to_oval_roi(ij_instance, image, oval_array: np.ndarray, add_to_roi_manager=False):
    # get ImageJ resources
    Dataset = sj.jimport('net.imagej.Dataset')
    ImagePlus = sj.jimport('ij.ImagePlus')
    OvalRoi = sj.jimport('ij.gui.OvalRoi')
    Overlay = sj.jimport('ij.gui.Overlay')
    ov = Overlay()

    if isinstance(image, Dataset):
        imp = ij_instance.convert().convert(image, ImagePlus)
    else:
        imp = image

    if add_to_roi_manager:
        RoiManager = sj.jimport('ij.plugin.frame.RoiManager')()
        rm = RoiManager.getRoiManager()

    for i in range(len(oval_array)):
        values = oval_array[i].tolist()
        y = values[0]
        x = values[1]
        r = values[2]
        d = r * 2
        roi = OvalRoi(x - r, y - r, d, d)
        imp.setRoi(roi)
        ov.add(roi)
        if add_to_roi_manager:
            rm.addRoi(roi)

    imp.setOverlay(ov)
    imp.show()

    return None