import matplotlib
import matplotlib.cm
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize
import torch.nn.functional as F


import skimage
from scipy import ndimage
from PIL import Image
# from pytorch3d.structures import Pointclouds



def nearest_neighbor_fill(img, mask, erosion=0):
    img_ = np.copy(img.cpu().numpy())

    if erosion > 0:
        eroded_mask = skimage.morphology.binary_erosion(mask.cpu().numpy(), footprint=skimage.morphology.disk(erosion))
    else:
        eroded_mask = mask.cpu().numpy()

    img_[eroded_mask <= 0] = np.nan

    distance_to_boundary = ndimage.distance_transform_bf((~eroded_mask>0), metric="cityblock")

    for current_dist in np.unique(distance_to_boundary)[1:]:
        ii, jj = np.where(distance_to_boundary == current_dist)

        ii_ = np.array([ii - 1, ii, ii + 1, ii - 1, ii, ii + 1, ii - 1, ii, ii + 1]).reshape(9, -1)
        jj_ = np.array([jj - 1, jj - 1, jj - 1, jj, jj, jj, jj + 1, jj + 1, jj + 1]).reshape(9, -1)

        ii_ = ii_.clip(0, img_.shape[0] - 1)
        jj_ = jj_.clip(0, img_.shape[1] - 1)

        img_[ii, jj] = np.nanmax(img_[ii_, jj_], axis=0)

    return torch.from_numpy(img_).to(img.device)


def snap_high_gradients_to_nn(depth, threshold=20):
    grad_depth = np.copy(depth.cpu().numpy())
    grad_depth = grad_depth - grad_depth.min()
    grad_depth = grad_depth / grad_depth.max()

    grad = skimage.filters.rank.gradient(grad_depth, skimage.morphology.disk(1))
    return nearest_neighbor_fill(depth, torch.from_numpy(grad < threshold).to(depth.device), erosion=3)






def project_points(cameras, depth, use_pixel_centers=True):
    if len(cameras) > 1:
        import warnings
        warnings.warn("project_points assumes only a single camera is used")

    depth_t = torch.from_numpy(depth) if isinstance(depth, np.ndarray) else depth
    depth_t = depth_t.to(cameras.device)

    pixel_center = 0.5 if use_pixel_centers else 0

    fx, fy = cameras.focal_length[0, 1], cameras.focal_length[0, 0]
    cx, cy = cameras.principal_point[0, 1], cameras.principal_point[0, 0]

    i, j = torch.meshgrid(
        torch.arange(cameras.image_size[0][0], dtype=torch.float32, device=cameras.device) + pixel_center,
        torch.arange(cameras.image_size[0][1], dtype=torch.float32, device=cameras.device) + pixel_center,
        indexing="xy",
    )

    directions = torch.stack(
        [-(i - cx) * depth_t / fx, -(j - cy) * depth_t / fy, depth_t], -1
    )

    xy_depth_world = cameras.get_world_to_view_transform().inverse().transform_points(directions.view(-1, 3)).unsqueeze(0)

    return xy_depth_world




# def get_pointcloud(xy_depth_world, device="cpu", features=None):
#     point_cloud = Pointclouds(points=[xy_depth_world.to(device)], features=[features] if features is not None else None)
#     return point_cloud

# def merge_pointclouds(point_clouds):
#     points = torch.cat([pc.points_padded() for pc in point_clouds], dim=1)
#     features = torch.cat([pc.features_padded() for pc in point_clouds], dim=1)
#     return Pointclouds(points=[points[0]], features=[features[0]])







def colorize(value, vmin=None, vmax=None, cmap='jet', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],98) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img

def save_point_cloud_as_ply(points, filename="output.ply", colors=None):
    """
    Save a PyTorch tensor of shape [N, 3] as a PLY file. Optionally with colors.
    
    Parameters:
    - points (torch.Tensor): The point cloud tensor of shape [N, 3].
    - filename (str): The name of the output PLY file.
    - colors (torch.Tensor, optional): The color tensor of shape [N, 3] with values in [0, 1]. Default is None.
    """
    
    # points = torch.from_numpy(points).permute(1,0)
    # colors = torch.from_numpy(colors)
    
    
    assert points.dim() == 2 and points.size(1) == 3, "Input tensor should be of shape [N, 3]."
    
    if colors is not None:
        assert colors.dim() == 2 and colors.size(1) == 3, "Color tensor should be of shape [N, 3]."
        assert points.size(0) == colors.size(0), "Points and colors tensors should have the same number of entries."
    
    # Header for the PLY file
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {points.size(0)}",
        "property float x",
        "property float y",
        "property float z"
    ]
    
    # Add color properties to header if colors are provided
    if colors is not None:
        header.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue"
        ])
    
    header.append("end_header")
    
    # Write to file
    with open(filename, "w") as f:
        for line in header:
            f.write(line + "\n")
        
        for i in range(points.size(0)):
            line = f"{points[i, 0].item()} {points[i, 1].item()} {points[i, 2].item()}"
            
            # Add color data to the line if colors are provided
            if colors is not None:
                # Scale color values from [0, 1] to [0, 255] and convert to integers
                r, g, b = (colors[i] * 255).clamp(0, 255).int().tolist()
                line += f" {r} {g} {b}"
            
            f.write(line + "\n")
            
            
