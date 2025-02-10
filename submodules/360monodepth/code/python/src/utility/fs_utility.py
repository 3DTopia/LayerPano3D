
from logging import root
import pathlib

import os

from logger import Logger

log = Logger(__name__)
log.logger.propagate = False


# The file name convention.
class FileNameConvention:
    """
    Visualized depth/disparity map is the end with *.pdf.jpg, having same file name with the file.
    The first bracket is the file name prefix.
    """

    def __init__(self):
        self.prefix_name = None
        self.root_dir = None

        # ERP image filename expression
        self.erp_dispmap_filename_expression                       = "{prefix_name}_disp.pfm"
        self.erp_dispmap_vis_filename_expression                   = "{prefix_name}_disp_vis.jpg"
        self.erp_depthmap_filename_expression                      = "{prefix_name}_depth.pfm"
        self.erp_depthmap_vis_filename_expression                  = "{prefix_name}_depth_vis.jpg"
        self.erp_depthmap_blending_result_filename_expression      = "{prefix_name}_depth_blending.pfm"
        self.erp_depthmap_vis_blending_result_filename_expression  = "{prefix_name}_depth_blending_vis.jpg" 

        # RGB image file name expression
        self.erp_rgb_filename_expression                           = "{}"
        self.subimage_rgb_filename_expression                      = "{prefix_name}_rgb_{:03d}.jpg"

        # disparity map file name expression
        self.subimage_dispmap_filename_expression                  = "{prefix_name}_disp_{:03d}.pfm"
        self.subimage_dispmap_aligned_filename_expression          = "{prefix_name}_disp_{:03d}_aligned.pfm"

        self.subimage_dispmap_persp_filename_expression            = "{prefix_name}_disp_persp_{:03d}.pfm"
        self.subimage_dispmap_persp_aligned_filename_expression    = "{prefix_name}_disp_persp_{:03d}_aligned.pfm"
        self.subimage_dispmap_erp_filename_expression              = "{prefix_name}_disp_erp_{:03d}.pfm"
        self.subimage_dispmap_erp_aligned_filename_expression      = "{prefix_name}_disp_erp_{:03d}_aligned.pfm"
 
        self.subimage_dispmap_aligning_filename_expression         = "{prefix_name}_disp_{:03d}_{}_aligning.pfm"
        self.subimage_dispmap_cpp_aligned_filename_expression      = "depthmapAlignPy_depth_{:03d}_aligned.pfm"

        # depth map file name expression
        self.subimage_depthmap_filename_expression                 = "{prefix_name}_depth_{:03d}.pfm"
        self.subimage_depthmap_aligned_filename_expression         = "{prefix_name}_depth_{:03d}_aligned.pfm"

        self.subimage_depthmap_persp_filename_expression           = "{prefix_name}_depth_persp_{:03d}.pfm"
        self.subimage_depthmap_persp_aligned_filename_expression   = "{prefix_name}_depth_persp_{:03d}_aligned.pfm"
        self.subimage_depthmap_erp_filename_expression             = "{prefix_name}_depth_erp_{:03d}.pfm"                  # the radius self.depth map
        self.subimage_depthmap_erp_aligned_filename_expression     = "{prefix_name}_depth_erp_{:03d}_aligned.pfm"

        # The intermedia file to debug the depthmap alignment process
        self.subimage_dispmap_aligned_coeffs_filename_expression   = "{prefix_name}_disp_coeff.json"                       # the depth map alignment coefficients
        self.subimage_pixelcorr_filename_expression                = "{prefix_name}_corr_{:03d}_{:03d}.json"               # the depth map alignment corresponding relationship file
        self.subimage_warpedimage_filename_expression              = "{prefix_name}_srcwarp_rgb_{:03d}_{:03d}_{}.jpg"
        self.subimage_warpeddepth_filename_expression              = "{prefix_name}_srcwarp_disp_{:03d}_{:03d}_{}.jpg"
        self.subimage_alignment_intermedia_filename_expression     = "{prefix_name}_alignment_{}.pickle"
        self.subimage_alignment_depthmap_input_filename_expression = "{prefix_name}_alignment_input_{}_{}.jpg"
        self.subimage_camparam_filename_expression                 = "{prefix_name}_cam_{:03d}.json" 
        self.subimage_camparam_list_filename_expression            = "{prefix_name}_cam_{:03d}.json"
        self.subimage_camsparams_list_filename_expression          = "{prefix_name}_cam_all.json"

    def set_filename_basename(self, prefix_name):
        """
        Set all filename expression's filename basename.
        """
        print("Set filename prefix: {}".format(prefix_name))
        self.prefix_name = prefix_name
        for attr in dir(self):
            if not callable(getattr(self, attr)) and not attr.startswith("__"):
                attr_value = getattr(self, attr)
                if attr_value is None:
                    continue
                newfilename = attr_value.replace("{prefix_name}", prefix_name)
                setattr(self, attr, newfilename)


    def set_filepath_folder(self, root_dir):
        """
        Set the file root folder.
        """
        print("Set file root dir: {}".format(root_dir))
        self.root_dir = root_dir
        for attr in dir(self):
            if not callable(getattr(self, attr)) and not attr.startswith("__"):
                attr_value = getattr(self, attr)
                if attr_value is None:
                    continue
                setattr(self, attr, os.path.join(root_dir, attr_value))


def dir_make(directory):
    """
    check the existence of directory, if not mkdir
    :param directory: the directory path
    :type directory: str
    """
    # check
    if isinstance(directory, str):
        directory_path = pathlib.Path(directory)
    elif isinstance(directory, pathlib.Path):
        directory_path = directory
    else:
        log.warn("Directory is neither str nor pathlib.Path {}".format(directory))
        return
    # create folder
    if not directory_path.exists():
        directory_path.mkdir()
    else:
        log.info("Directory {} exist".format(directory))


def dir_ls(dir_path, postfix = None):
    """Find all files in a directory with extension.

    :param dir_path: folder path.
    :type dir_path: str
    :param postfix: extension, e.g. ".txt", if it's none list all folders name.
    :type postfix: str
    """
    file_list = []
    for file_name in os.listdir(dir_path):
        if os.path.isdir(dir_path + "/" + file_name) and postfix is None:
            file_list.append(file_name)
        elif postfix is not None:
            if file_name.endswith(postfix):
                file_list.append(file_name)
    file_list.sort()
    return file_list


def dir_rm(dir_path):
    """Deleting folders recursively.

    :param dir_path: The folder path.
    :type dir_path: str
    """
    directory = pathlib.Path(dir_path)
    if not directory.exists():
        log.warn("Directory {} do not exist".format(dir_path))
        return
    for item in directory.iterdir():
        if item.is_dir():
            dir_rm(item)
        else:
            item.unlink()
    directory.rmdir()


def exist(path, dest_type=0):
    """File exist.

    :param path: [description]
    :type path: [type]
    :param dest_type: 1 is file, 2 is directory, 0 is both.
    :type dest_type: int
    """
    if dest_type == 1:
        return os.path.isfile(path)
    elif dest_type == 2:
        return os.path.isdir(path)
    elif dest_type == 0:
        return os.path.exists(path)


def file_rm(path):
    """Remove a file.

    :param path: [description]
    :type path: str
    """
    if not exist(path, 1):
        log.debug("{} do not exist.".format(path))
        return
    elif exist(path, 1):
        os.remove(path)
    elif exist(path, 2):
        log.warn("{} is a folder".format(path))

