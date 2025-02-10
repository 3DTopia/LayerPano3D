import os
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt

from logger import Logger
from utility import depthmap_utils

log = Logger(__name__)
log.logger.propagate = False


def subimage_save_ico(subimage_list_, output_path, subimage_idx_list=None):
    """save the visualized depth map array to image file with value-bar.

    :param dapthe_data: The depth data.
    :type dapthe_data: numpy 
    :param output_path: the absolute path of output image.
    :type output_path: str
    :param subimage_idx_list: available subimages index list.
    :type subimage_idx_list: list
    """
    # add blank image to miss subimage to fill the sub-image array
    subimage_list = None
    if len(subimage_list_) != 20 \
            and subimage_idx_list is not None \
            and len(subimage_list_) == len(subimage_idx_list):
        log.debug("The ico's sub-image size is {}, fill blank sub-images.".format(len(subimage_list_)))
        subimage_list = [np.zeros_like(subimage_list_[0])] * 20
        for subimage_index in range(len(subimage_idx_list)):
            subimage_face_idx = subimage_idx_list[subimage_index]
            subimage_list[subimage_face_idx] = subimage_list_[subimage_index]
    elif len(subimage_list_) == 20:
        subimage_list = subimage_list_
    else:
        raise log.error("The sub-image is not completed.")

    # draw image
    figure, axes = plt.subplots(4, 5)
    counter = 0
    for row_index in range(0, 4):
        for col_index in range(0, 5):
            axes[row_index, col_index].get_xaxis().set_visible(False)
            axes[row_index, col_index].get_yaxis().set_visible(False)
            # add sub caption
            axes[row_index, col_index].set_title(str(counter))
            counter = counter + 1
            #
            dispmap_index = row_index * 5 + col_index
            im = axes[row_index, col_index].imshow(subimage_list[dispmap_index].astype(np.uint8))

    figure.tight_layout()
    # plt.colorbar(im, ax=axes.ravel().tolist())
    plt.savefig(output_path, dpi=150)
    # plt.show()
    plt.close(figure)


def image_read(image_file_path):
    """[summary]

    :param image_file_path: the absolute path of image
    :type image_file_path: str
    :return: the numpy array of image
    :rtype: numpy
    """    
    if not os.path.exists(image_file_path):
        log.error("{} do not exist.".format(image_file_path))

    return np.asarray(Image.open(image_file_path))


def image_show(image, title=" ",  verbose=True):
    """
    visualize the numpy array
    """
    if len(np.shape(image)) == 3:
        print("show 3 channels rgb image")
        image_rgb = image.astype(int)
        plt.title(title)
        plt.axis("off")
        plt.imshow(image_rgb)
        plt.show()
    elif len(np.shape(image)) == 2:
        print("visualize 2 channel raw data")
        images = []
        cmap = plt.get_cmap('rainbow')
        fig, axs = plt.subplots(nrows=1, sharex=True, figsize=(3, 5))
        axs.set_title(title)
        images.append(axs.imshow(image, cmap=cmap))
        fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1, shrink=0.4)
        plt.show()
    elif len(np.shape(image)) == 1:
        print("show 1 channels data array")
        image_rgb = visual_data(image, verbose=False)
        plt.title(title)
        plt.axis("off")
        plt.imshow(image_rgb)
        plt.show()

    else:
        print("the data channel is {}, should be visualized in advance.".format(len(np.shape(image))))

def image_save(image_data, image_file_path):
    """Save numpy array as image.

    :param image_data: Numpy array store image data. numpy 
    :type image_data: numpy
    :param image_file_path: The image's path
    :type image_file_path: str
    """
    # 0) convert the datatype
    image = None
    if image_data.dtype in [float, np.int64, int]:
        print("saved image array type is {}, converting to uint8".format(image_data.dtype))
        image = image_data.astype(np.uint8)
    else:
        image = image_data

    # 1) save to image file
    image_channels_number = image.shape[2]
    if image_channels_number == 3:
        im = Image.fromarray(image)
        im.save(image_file_path)
    else:
        log.error("The image channel number is {}".format(image_channels_number))


def image_show_pyramid(subimage_list):
    """save the visualized depth map array to image file with value-bar.

    :param dapthe_data: The depth data.
    :type dapthe_data: numpy 
    :param output_path: the absolute path of output image.
    :type output_path: str
    :param subimage_idx_list: available subimages index list.
    :type subimage_idx_list: list
    """
    # add blank image to miss subimage to fill the sub-image array
    image_number = len(subimage_list)
    pyramid_depth = len(subimage_list[0])

    # draw image
    figure, axes = plt.subplots(image_number, pyramid_depth)
    for image_idx in range(0,image_number):
        for pyramid_idx in range(0, pyramid_depth):
            axes[image_idx, pyramid_idx].get_xaxis().set_visible(False)
            axes[image_idx, pyramid_idx].get_yaxis().set_visible(False)
            #
            dispmap_vis = depthmap_utils.depth_visual(subimage_list[pyramid_idx][image_idx])
            # add sub caption
            image_size_str = "Idx:{}, Level:{}, {}x{}".format(image_idx, pyramid_idx, dispmap_vis.shape[0], dispmap_vis.shape[1])
            axes[image_idx, pyramid_idx].set_title(image_size_str)
            im = axes[image_idx, pyramid_idx].imshow(dispmap_vis)#.astype(np.uint8)

    figure.tight_layout()
    plt.show()
