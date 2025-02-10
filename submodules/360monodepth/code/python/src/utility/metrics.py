import os.path

import numpy as np
from PIL import Image

from logger import Logger
import depthmap_utils

log = Logger(__name__)
log.logger.propagate = False


# ==========================
# Depth Prediction Metrics
# Refernece
# - zioulis2018omnidepth
# ==========================
eps = 1e-7


def abs_rel_error(pred, gt, mask):
    """Compute absolute relative difference error"""
    return np.mean(np.abs(pred[mask > 0] - gt[mask > 0]) / np.maximum(np.abs(gt[mask > 0]),
                                                                      np.full_like(gt[mask > 0], eps)))


def abs_rel_error_map(pred, gt, mask):
    """ per pixels' absolute relative difference.

    Parameters @see delta_inlier_ratio_map
    :return: invalid pixel is NaN
    """
    are_map = np.zeros_like(pred)
    are_map[mask > 0] = np.abs(pred[mask > 0] - gt[mask > 0]) / gt[mask > 0]
    are_map[mask <= 0] = np.nan
    return are_map


def sq_rel_error(pred, gt, mask):
    """Compute squared relative difference error"""
    return np.mean((pred[mask > 0] - gt[mask > 0]) ** 2 / np.maximum(np.abs(gt[mask > 0]),
                                                                     np.full_like(gt[mask > 0], eps)))


def sq_rel_error_map(pred, gt, mask):
    """ squared relative difference error map.
    Parameters @see delta_inlier_ratio_map
    """
    are_map = np.zeros_like(pred)
    are_map[mask > 0] = (pred[mask > 0] - gt[mask > 0]) ** 2 / gt[mask > 0]
    are_map[mask <= 0] = np.nan
    return are_map


def mean_absolute_error(pred, gt, mask):
    """Mean absolute error"""
    return np.mean(np.abs(pred[mask > 0] - gt[mask > 0]))


def lin_rms_sq_error(pred, gt, mask):
    """Compute the linear RMS error except the final square-root step"""
    return np.mean((pred[mask > 0] - gt[mask > 0]) ** 2)


def lin_rms_sq_error_map(pred, gt, mask):
    """ Each pixel RMS.
    """
    lin_rms_map = np.zeros_like(pred)
    lin_rms_map[mask > 0] = (pred[mask > 0] - gt[mask > 0]) ** 2
    lin_rms_map[mask <= 0] = np.nan
    return lin_rms_map


def log_rms_sq_error(pred, gt, mask):
    """Compute the log RMS error except the final square-root step"""
    # if np.any(pred[mask] < 0) or np.any(gt[mask] < 0):
    #     log.error("The disparity map has negative value! The metric log will generate NaN")

    mask = (mask > 0) & (pred > eps) & (gt > eps)  # Compute a mask of valid values
    return np.mean((np.log10(pred[mask]) - np.log10(gt[mask])) ** 2)


def log_rms_sq_error_map(pred, gt, mask):
    """ Each pixel log RMS.
    Parameters @see delta_inlier_ratio_map
    """
    # if np.any(pred[mask] < 0) or np.any(gt[mask] < 0):
    #     log.error("The disparity map has negative value! The metric log will generate NaN")
    mask = (mask > 0) & (pred > eps) & (gt > eps)  # Compute a mask of valid values

    log_rms_map = np.zeros_like(pred)
    log_rms_map[mask > 0] = (np.log10(pred[mask > 0]) - np.log10(gt[mask > 0])) ** 2 / gt[mask > 0]
    log_rms_map[mask <= 0] = np.nan
    return log_rms_map


def log_rms_scale_invariant(pred, gt, mask):
    """ scale-invariant log RMSE.
    """
    if np.any(pred[mask] < 0) or np.any(gt[mask] < 0):
        log.error("The disparity map has negative value! The metric log will generate NaN")

    alpha_depth = np.mean(np.log(pred[mask > 0]) - np.log(gt[mask > 0]))
    log_rms_scale_inv = np.mean(np.log(pred[mask > 0]) - np.log(gt[mask > 0]) + alpha_depth)
    return log_rms_scale_inv


def log_rms_scale_invariant_map(pred, gt, mask):
    """ Each pixel scale invariant log RMS.
    Parameters @see delta_inlier_ratio_map
    """
    if np.any(pred[mask] < 0) or np.any(gt[mask] < 0):
        log.error("The disparity map has negative value! The metric log will generate NaN")

    log_rms_map = np.zeros_like(pred)
    alpha_depth = np.mean(np.log(pred[mask > 0]) - np.log(gt[mask > 0]))
    log_rms_map[mask > 0] = np.log(pred[mask > 0]) - np.log(gt[mask > 0]) + alpha_depth
    log_rms_map[mask <= 0] = np.nan
    return log_rms_map


def delta_inlier_ratio(pred, gt, mask, degree=1):
    """Compute the delta inlier rate to a specified degree (def: 1)"""
    return np.mean(np.maximum(pred[mask > 0] / gt[mask > 0], gt[mask > 0] / pred[mask > 0]) < (1.25 ** degree))


def delta_inlier_ratio_map(pred, gt, mask, degree=1):
    """ Get the δ < 1.25^degree map.

    Get the δ map, if pixels less than thr is 1, larger is 0, invalid is -1.

    :param pred: predict disparity map, [height, width]
    :type pred: numpy
    :param gt: ground truth disparity map, [height, width]
    :type gt: numpy
    :param mask: If the mask is greater than 0 the pixel is available, otherwise it's invalided.
    :type mask: numpy
    :param degree: The exponent of 1.24, defaults to 1
    :type degree: int, optional
    :return: The δ map, [height, width]
    :rtype: numpy
    """
    delta_max = np.maximum(pred[mask > 0] / gt[mask > 0], gt[mask > 0] / pred[mask > 0])

    delta_map = np.zeros_like(delta_max)
    delta_less = delta_max < (1.25 ** degree)
    delta_map[delta_less] = 1

    delta_larger = delta_max >= (1.25 ** degree)
    delta_map[delta_larger] = 0

    delta_map_all = np.zeros_like(pred)
    delta_map_all[mask > 0] = delta_map
    delta_map_all[mask <= 0] = -1
    return delta_map_all


def normalize_depth_maps(pred, gt, mask):

    # Compute median and substract
    median_gt = np.median(gt[mask])
    median_pred = np.median(pred[mask])
    sub_med_pred = pred - median_pred

    #   Get the deviation of the valid pixels
    dev_gt = np.sum(np.abs(gt[mask] - median_gt)) / np.sum(mask)
    dev_pred = np.sum(np.abs(pred[mask] - median_pred)) / np.sum(mask)

    pred = sub_med_pred / dev_pred * dev_gt + median_gt

    return gt, pred


def pred2gt_least_squares(pred, gt, mask, max_depth=10):
    gt = depthmap_utils.depth2disparity(gt)
    a_00 = np.sum(pred[mask] * pred[mask])
    a_01 = np.sum(pred[mask])
    a_11 = np.sum(mask)

    b_0 = np.sum(pred[mask] * gt[mask])
    b_1 = np.sum(gt[mask])

    det = a_00 * a_11 - a_01 * a_01

    s = (a_11 * b_0 - a_01 * b_1) / det
    o = (-a_01 * b_0 + a_00 * b_1) / det

    pred = s * pred + o
    pred = depthmap_utils.disparity2depth(pred)
    # return pred
    return np.clip(pred, 0, max_depth)


def report_error(gt, pred, max_depth=10.0):
    mask = (gt > 0) & (~np.isinf(gt)) & (~np.isnan(gt)) & (gt <= max_depth)

    pred = pred2gt_least_squares(pred, gt, mask)

    metrics_res = {"AbsRel": abs_rel_error(pred, gt, mask),
                   "SqRel": sq_rel_error(pred, gt, mask),
                   "MAE": mean_absolute_error(pred, gt, mask),
                   "RMSE": np.sqrt(lin_rms_sq_error(pred, gt, mask)),
                   "RMSELog": np.sqrt(log_rms_sq_error(pred, gt, mask)),
                   "Delta1": delta_inlier_ratio(pred, gt, mask, degree=1),
                   "Delta2": delta_inlier_ratio(pred, gt, mask, degree=2),
                   "Delta3": delta_inlier_ratio(pred, gt, mask, degree=3)}

    return metrics_res


def normalize_depth_maps2(pred, gt, mask):
    median_gt = np.median(gt[mask])
    median_pred = np.median(pred[mask])

    pred *= median_gt/median_pred

    return gt, pred


def visualize_error_maps(pred, gt, mask, idx=0, save=False, input=None, filename=""):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import gc
    fig = plt.figure(figsize=(9.24, 9.82))

    output_path = "../../../results/"
    if os.path.dirname(filename) != '':
        dir_name = os.path.dirname(filename)
        filename = os.path.basename(filename)
        output_path = os.path.join(output_path, dir_name)
        os.makedirs(output_path, exist_ok=True)

    input_filename = os.path.join(output_path, '{:04}.png'.format(idx))
    error_map_filename = os.path.join(output_path, '{:04}_{}_error.png'.format(idx, filename)) if filename != "" else \
        os.path.join(output_path, '{:04}_error.png'.format(idx))
    i = 0
    if input is not None and not os.path.isfile(input_filename):
        Image.fromarray(np.uint8(input)).save(input_filename)

    if os.path.isfile(error_map_filename):
        return

    gs = fig.add_gridspec(5, 2)
    ax = fig.add_subplot(gs[i, 0])
    ax.axis("off")
    ax.title.set_text("GT")
    im0 = ax.imshow(gt, cmap="turbo", vmin=0, vmax=10)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')
    ax = fig.add_subplot(gs[i, 1])
    ax.axis("off")
    ax.title.set_text("Pred")
    im0 = ax.imshow(pred, cmap="turbo", vmin=0, vmax=10)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')

    i += 1
    ax = fig.add_subplot(gs[i, 0])
    ax.axis("off")
    ax.title.set_text("Abs_rel")
    im0 = ax.imshow(abs_rel_error_map(pred, gt, mask), cmap="RdPu")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')
    ax = fig.add_subplot(gs[i, 1])
    ax.axis("off")
    ax.title.set_text("Sq_rel")
    im0 = ax.imshow(sq_rel_error_map(pred, gt, mask), cmap="RdPu")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')

    i += 1
    ax = fig.add_subplot(gs[i, 0])
    ax.axis("off")
    ax.title.set_text("RMS")
    im0 = ax.imshow(lin_rms_sq_error_map(pred, gt, mask), cmap="RdPu")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')
    ax = fig.add_subplot(gs[i, 1])
    ax.axis("off")
    ax.title.set_text("RMS(log)")
    im0 = ax.imshow(log_rms_sq_error_map(pred, gt, mask), cmap="RdPu")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')

    i+=1
    ax = fig.add_subplot(gs[i, 0])
    ax.axis("off")
    ax.title.set_text("Delta 1")
    im0 = ax.imshow(delta_inlier_ratio_map(pred, gt, mask, 1), cmap="RdPu")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')
    ax = fig.add_subplot(gs[i, 1])
    ax.axis("off")
    ax.title.set_text("Delta 2")
    im0 = ax.imshow(delta_inlier_ratio_map(pred, gt, mask, 2), cmap="RdPu")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')

    i += 1
    ax = fig.add_subplot(gs[i, 0])
    ax.axis("off")
    ax.title.set_text("Delta 3")
    im0 = ax.imshow(delta_inlier_ratio_map(pred, gt, mask, 3), cmap="RdPu")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')
    ax = fig.add_subplot(gs[i, 1])
    ax.axis("off")
    abs_dif = np.zeros_like(pred)
    abs_dif[mask > 0] = np.abs(pred[mask > 0] - gt[mask > 0])
    abs_dif[mask <= 0] = np.nan
    ax.title.set_text("|GT-Pred|")
    im0 = ax.imshow(abs_dif, cmap="RdPu")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')
    plt.tight_layout(pad=0.1, h_pad=-0.5, w_pad=3)
    if save:
        plt.savefig(error_map_filename, dpi=150)
    plt.cla()
    plt.clf()
    plt.close('all')
    plt.close(fig)
    gc.collect()
