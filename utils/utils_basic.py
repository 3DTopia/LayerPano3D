from PIL import Image
import numpy as np


def icosahedron_sample_camera(auto_resort=False):
    # reference: https://en.wikipedia.org/wiki/Regular_icosahedron
    radius_circumscribed = np.sin(2 * np.pi / 5.0)
    radius_inscribed = np.sqrt(3) / 12.0 * (3 + np.sqrt(5))
    radius_midradius = np.cos(np.pi / 5.0)
    theta_step = 2.0 * np.pi / 5.0

    thetas = []
    phis = []
    for triangle_index in range(20):
        # 1) the up 5 triangles
        if 0 <= triangle_index <= 4:
            theta = - np.pi + theta_step / 2.0 + triangle_index * theta_step
            phi = np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed)

        # 2) the middle 10 triangles
        # 2-0) middle-up triangles
        if 5 <= triangle_index <= 9:
            triangle_index_temp = triangle_index - 5
            theta = - np.pi + theta_step / 2.0 + triangle_index_temp * theta_step
            phi = np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius)

        # 2-1) the middle-down triangles
        if 10 <= triangle_index <= 14:
            triangle_index_temp = triangle_index - 10
            theta = - np.pi + triangle_index_temp * theta_step
            phi = -(np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius))

        # 3) the down 5 triangles
        if 15 <= triangle_index <= 19:
            triangle_index_temp = triangle_index - 15
            theta = - np.pi + triangle_index_temp * theta_step
            phi = - (np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed))

        thetas.append(theta)
        phis.append(phi)

    if auto_resort:
        thetas = thetas[5:] + thetas[:5]
        phis = phis[5:] + phis[:5]

    thetas, phis = np.rad2deg(np.array(thetas)), np.rad2deg(np.array(phis))

    return thetas, phis  #degree
    ## theta:[-144.  -72.    0.   72.  144. -144.  -72.    0.   72.  144. -180. -108.  -36.   36.  108. -180. -108.  -36.   36.  108.]
    ## phi: [ 52.62263186  52.62263186  52.62263186  52.62263186  52.62263186
            # 10.81231696  10.81231696  10.81231696  10.81231696  10.81231696
            # -10.81231696 -10.81231696 -10.81231696 -10.81231696 -10.81231696
            # -52.62263186 -52.62263186 -52.62263186 -52.62263186 -52.62263186]
            
def get_bagel_mask(mask_img):
    mask = np.asarray(mask_img) 
    mask = mask / mask.max()
    h, w = mask.shape
    new_mask = mask.copy()
    filter_size = 11
    for i in range(0,h-filter_size,5):
        for j in range(0,w-filter_size,5):
            if np.sum(mask[i:i+filter_size, j:j+filter_size]) < filter_size**2:
                small_mask = mask[i:i+filter_size, j:j+filter_size] > 0.5
                new_small_mask = new_mask[i:i+filter_size, j:j+filter_size]
                new_small_mask[small_mask] = 0.5
                new_mask[i:i+filter_size, j:j+filter_size] = new_small_mask

    return new_mask

def get_bagel_mask2(mask_smooth, mask_sharp):
    mask_smooth = np.asarray(mask_smooth) 
    mask_smooth = mask_smooth / mask_smooth.max()
    mask_sharp = np.asarray(mask_sharp) 
    mask_sharp = mask_sharp / mask_sharp.max()
    mask_diff = np.logical_xor(mask_smooth.astype(bool), mask_sharp.astype(bool))
    mask_bagel = mask_smooth.copy().astype(float)
    mask_bagel[mask_diff] = 0.5
  
    return mask_bagel



def make_bagel_mask(mask_path, mask_smooth_path):

    mask = Image.open(mask_path).convert('L')
    mask2 = Image.open(mask_smooth_path).convert('L')

    mask_bagel_shape = get_bagel_mask(mask)
    mask_bagel = get_bagel_mask2(mask2, mask)
    mask_bagel[np.asarray(mask).astype(bool)] = mask_bagel_shape[np.asarray(mask).astype(bool)] #(h, w) float64 [0-1]
    
    # mask_bagel = Image.fromarray((mask_bagel*255).astype(np.uint8)) #PIL uint8 [0-255]
    mask_bagel = mask_bagel[..., None]
    return mask_bagel ## return (h, w, 1) float64 [0-1]