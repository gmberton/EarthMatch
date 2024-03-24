
import cv2
import math
import torch
import numpy as np
from PIL import Image
from lightglue import viz2d
import matplotlib.pyplot as plt
import torchvision.transforms as tfm
from torchvision.transforms.functional import InterpolationMode


def path_to_footprint(path):
    _, min_lat, min_lon, _, _, max_lat, max_lon = str(path).split("@")[:7]
    min_lat, min_lon, max_lat, max_lon = float(min_lat), float(min_lon), float(max_lat), float(max_lon)
    pred_footprint = np.array([min_lat, min_lon, max_lat, min_lon, max_lat, max_lon, min_lat, max_lon])
    pred_footprint = pred_footprint.reshape(4, 2)
    pred_footprint = torch.tensor(pred_footprint)
    return pred_footprint


def px_to_coord(HW, footprint, pxx, pxy):
    lat_per_pixel, lon_per_pixel = get_lat_lon_per_pixel(footprint, HW)
    min_lat, min_lon, max_lat, max_lon = footprint_to_minmax_latlon(footprint)
    lat = max_lat - (pxy * lat_per_pixel)
    lon = min_lon + (pxx * lon_per_pixel)
    return float(lat), float(lon)

def coord_to_px(HW, footprint, lat, lon):
    lat_per_pixel, lon_per_pixel = get_lat_lon_per_pixel(footprint, HW)
    min_lat, min_lon, max_lat, max_lon = footprint_to_minmax_latlon(footprint)
    pxy = (max_lat - lat) / lat_per_pixel
    pxx = (lon - min_lon) / lon_per_pixel
    return float(pxx), float(pxy)


def rotate_footprint(pred_footprint, pred_rot_angle):
    num_rots = (pred_rot_angle // 90) + 1
    return np.array([pred_footprint[i%4] for i in range(num_rots, num_rots+4)])


# Function to compute the transformed corners using a homography matrix
def transform_corners(width, height, homography):
    """
    Transform the four corners of an image of given width and height using the provided homography matrix.
    :param width: Width of the image.
    :param height: Height of the image.
    :param homography: Homography matrix.
    :return: Transformed coordinates of the four corners.
    """
    # Define the four corners of the image
    corners = np.array([
        [0, 0],  # Top-left corner
        [width, 0],  # Top-right corner
        [width, height],  # Bottom-right corner
        [0, height]  # Bottom-left corner
    ], dtype='float32')
    # Reshape the corners for homography transformation
    corners = np.array([corners])
    corners = np.reshape(corners, (4, 1, 2))
    # Use the homography matrix to transform the corners
    transformed_corners = cv2.perspectiveTransform(corners, homography)
    return torch.tensor(transformed_corners).type(torch.int)[:, 0]


def tile_to_lat_lon_corners(zoom, row, col):
    num_tiles = 2 ** zoom
    tile_lon_width = 360.0 / num_tiles
    min_lon = col * tile_lon_width - 180.0
    max_lon = (col + 1) * tile_lon_width - 180.0
    max_lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * row / num_tiles))))
    min_lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (row + 1) / num_tiles))))
    return min_lat, max_lat, min_lon, max_lon


def get_image_with_neighbors(db_img_path, d_zrc_path, HW):
    """Return PIL image with surroundings of chosen image"""
    zoom, row, col = db_img_path.name.split("@")[9].split("_")
    zoom, row, col = [int(e) for e in [zoom, row, col]]
    min_lat1, max_lat1, min_lon1, max_lon1 = tile_to_lat_lon_corners(zoom, row+7, col-4)
    min_lat2, max_lat2, min_lon2, max_lon2 = tile_to_lat_lon_corners(zoom, row-4, col+7)
    min_lat = min(min_lat1, min_lat2)
    max_lat = max(max_lat1, max_lat2)
    min_lon = min(min_lon1, min_lon2)
    max_lon = max(max_lon1, max_lon2)
    lat_per_pixel = (max_lat - min_lat) / (HW * 3)
    lon_per_pixel = (max_lon - min_lon) / (HW * 3)
    center_min_lat = min_lat + HW * lat_per_pixel
    center_max_lat = min_lat + HW * lat_per_pixel * 2
    center_min_lon = min_lon + HW * lon_per_pixel
    center_max_lon = min_lon + HW * lon_per_pixel * 2
    
    surrounding_img = np.zeros([HW*3, HW*3, 3], dtype=np.uint8)
    surrounding_img[:, :] = 24, 43, 75
    for r in range(-1, 2, 1):
        for c in range(-1, 2, 1):
            try:
                p = d_zrc_path[f"{zoom:02d}_{row+r*4:04d}_{col+c*4:04d}"]
                if c == r == 0:
                    assert p == str(db_img_path), f"{p}\n{db_img_path}"
                img = np.array(Image.open(p))
                surrounding_img[(r+1)*HW : (r+2)*HW, (c+1)*HW : (c+2)*HW] = img
            except KeyError:
                pass
    surrounding_footprint = np.array([[min_lat, min_lon], [max_lat, min_lon], [max_lat, max_lon], [min_lat, max_lon]])
    center_footprint = np.array([[center_min_lat, center_min_lon], [center_max_lat, center_min_lon],
                                 [center_max_lat, center_max_lon], [center_min_lat, center_max_lon]])
    return tfm.Resize(HW*3)(Image.fromarray(surrounding_img)), \
        min_lat, min_lon, max_lat, max_lon, \
        center_min_lat, center_min_lon, center_max_lat, center_max_lon, \
        surrounding_footprint, center_footprint


def batch_geodesic_distances(origin, destination):
    assert type(origin) == type(destination) == torch.Tensor
    assert origin.shape[1] == destination.shape[1] == 2
    radius = 6371 # km
    lat1, lon1 = origin.T
    lat2, lon2 = destination.T
    dlat = torch.deg2rad(lat2-lat1)
    dlon = torch.deg2rad(lon2-lon1)
    a = torch.sin(dlat/2) * torch.sin(dlat/2) + torch.cos(torch.deg2rad(lat1)) \
        * torch.cos(torch.deg2rad(lat2)) * torch.sin(dlon/2) * torch.sin(dlon/2)
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
    distances = radius * c
    return distances


def compute_matching(image0, image1, matcher, save_images=False, viz_params=None):
    
    num_inliers, fm, mkpts0, mkpts1 = matcher(image0, image1)
    if num_inliers == 0:
        # The matcher did not find enough matches between img0 and img1
        return num_inliers, fm
    
    if save_images:
        path0 = viz_params["query_path"]
        path1 = viz_params["pred_path"]
        output_dir = viz_params["output_dir"]
        output_file_suffix = viz_params["output_file_suffix"]
        stem0, stem1 = path0.stem, path1.stem
        matches_path = output_dir / f'{stem0}_{stem1}_matches_{output_file_suffix}.torch'
        viz_path = output_dir / f'{stem0}_{stem1}_matches_{output_file_suffix}.jpg'
        output_dir.mkdir(exist_ok=True)
        viz2d.plot_images([image0, image1])
        viz2d.plot_matches(mkpts0, mkpts1, color="lime", lw=0.2)
        viz2d.add_text(0, f'{len(mkpts1)} matches', fs=20)
        viz2d.save_plot(viz_path)
        plt.close()
        torch.save((num_inliers, fm, mkpts0, mkpts1), matches_path)
    
    return num_inliers, fm


def footprint_to_minmax_latlon(footprint):
    lats = footprint[:, 0]
    lons = footprint[:, 1]
    min_lat = lats.min()
    max_lat = lats.max()
    min_lon = lons.min()
    max_lon = lons.max()
    return min_lat, min_lon, max_lat, max_lon

def get_lat_lon_per_pixel(footprint, HW):
    """Return the change in lat lon per each pixel"""
    min_lat, min_lon, max_lat, max_lon = footprint_to_minmax_latlon(footprint)
    lat_per_pixel = (max_lat - min_lat) / HW
    lon_per_pixel = (max_lon - min_lon) / HW
    return lat_per_pixel, lon_per_pixel


def warp_footprint_equirectangular_image(pred_footprint, transformed_corners, HW):
    lat_per_pixel, lon_per_pixel = get_lat_lon_per_pixel(pred_footprint, HW)
    min_lat, min_lon, max_lat, max_lon = footprint_to_minmax_latlon(pred_footprint)
    
    px_lats = transformed_corners[:, 1]
    px_lons = transformed_corners[:, 0]
    
    ul_lat = max_lat - (px_lats[0] * lat_per_pixel)
    ul_lon = min_lon + (px_lons[0] * lon_per_pixel)
    
    ur_lat = max_lat - (px_lats[1] * lat_per_pixel)
    ur_lon = min_lon + (px_lons[1] * lon_per_pixel)
    
    ll_lat = max_lat - (px_lats[2] * lat_per_pixel)
    ll_lon = min_lon + (px_lons[2] * lon_per_pixel)
    
    lr_lat = max_lat - (px_lats[3] * lat_per_pixel)
    lr_lon = min_lon + (px_lons[3] * lon_per_pixel)
    
    warped_pred_footprint = torch.tensor([
        [ul_lat, ul_lon], [ur_lat, ur_lon], [ll_lat, ll_lon], [lr_lat, lr_lon]
    ])
    return warped_pred_footprint


def add_homographies_fm(fm1, fm2):
    return np.linalg.inv(np.linalg.inv(fm2) @ np.linalg.inv(fm1))


def estimate_footprint(
    fm, query_image, surrounding_image, matcher,  surrounding_img_footprint, HW,
    save_images=False, viz_params=None
):
    """
    Parameters
    ----------
    fm : fundamental matrix from previous iteration (None if first iteration).
    query_image : torch.tensor with the query image
    surrounding_image : TYPE
        DESCRIPTION.
    matcher : TYPE
        DESCRIPTION.
    surrounding_img_footprint : TYPE
        DESCRIPTION.
    HW : TYPE
        DESCRIPTION.
    save_images : TYPE, optional
        DESCRIPTION. The default is False.
    viz_params : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    num_inliers : TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    assert surrounding_image.shape[1] == surrounding_image.shape[2], f"{surrounding_image.shape}"
    
    if fm is not None:
        transformed_corners = transform_corners(HW, HW, fm) + HW
        endpoints = [[HW, HW], [HW*2, HW], [HW*2, HW*2], [HW, HW*2]]
        warped_surrounding_pred_img = tfm.functional.perspective(
            surrounding_image, transformed_corners.numpy(), endpoints, InterpolationMode.BILINEAR
        )
        warped_pred_footprint = warp_footprint_equirectangular_image(surrounding_img_footprint, transformed_corners, HW*3)
    else:
        warped_surrounding_pred_img = surrounding_image
        warped_pred_footprint = surrounding_img_footprint
    
    assert tuple(warped_surrounding_pred_img.shape) == (3, HW*3, HW*3)
    warped_pred_img = warped_surrounding_pred_img[:, HW:HW*2, HW:HW*2]
    
    if save_images:
        tfm.ToPILImage()(warped_pred_img).save(viz_params["pred_path"])
    
    # path0 = viz_params["path0"]
    # path1 = viz_params["path1"]
    # output_dir = viz_params["output_dir"]
    # output_file_suffix = viz_params["output_file_suffix"]
        # path0=query_path, path1=pred_path,
        # output_dir=output_dir, output_file_suffix=f"{iteration}"
    
    num_inliers, new_fm = compute_matching(
        image0=query_image, image1=warped_pred_img,
        matcher=matcher, save_images=save_images,
        viz_params=viz_params
    )
    
    if num_inliers == 0:
        return num_inliers, None, None, None
    
    if fm is None:  # At first iteration fm is None
        fm = new_fm
    else:
        fm = add_homographies_fm(fm, new_fm)
    
    pretty_printed_footprint = "; ".join([f"{lat_lon[0]:.5f}, {lat_lon[1]:.5f}" for lat_lon in warped_pred_footprint])
    return num_inliers, fm, warped_pred_footprint, pretty_printed_footprint


