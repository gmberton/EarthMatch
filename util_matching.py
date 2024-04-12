
import cv2
import torch
import shapely
import numpy as np
from lightglue import viz2d
import matplotlib.pyplot as plt
import torchvision.transforms as tfm
from sklearn.linear_model import LogisticRegression
from torchvision.transforms.functional import InterpolationMode


def path_to_footprint(path):
    _, min_lat, min_lon, _, _, max_lat, max_lon = str(path).split("@")[:7]
    min_lat, min_lon, max_lat, max_lon = float(min_lat), float(min_lon), float(max_lat), float(max_lon)
    pred_footprint = np.array([min_lat, min_lon, max_lat, min_lon, max_lat, max_lon, min_lat, max_lon])
    pred_footprint = pred_footprint.reshape(4, 2)
    pred_footprint = torch.tensor(pred_footprint)
    return pred_footprint


def apply_homography_to_corners(width, height, homography):
    """
    Transform the four corners of an image of given width and height using the provided homography matrix.
    :param width: Width of the image.
    :param height: Height of the image.
    :param homography: Homography matrix.
    :return: Transformed pixel coordinates of the four corners.
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


def apply_homography_to_footprint(pred_footprint, transformed_corners, HW):

    # TODO these 6 lines are because the surrounding_pred_footprint is wrong, and it is actually the pred_footprint.
    center = pred_footprint.mean(0)
    diff = pred_footprint.max(0)[0] - pred_footprint.min(0)[0]
    diff *= 1.5
    min_lat, min_lon = center - diff
    max_lat, max_lon = center + diff
    pred_footprint = np.array([min_lat, min_lon, max_lat, min_lon, max_lat, max_lon, min_lat, max_lon]).reshape(4, 2)

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


def compute_threshold(true_matches, false_matches, thresh=0.999):
    assert isinstance(true_matches, list)
    assert isinstance(false_matches, list)
    if (len(true_matches) < 4):
        return 4
    # logistic_model = lambda x: 1 / (1 + np.exp(-x))
    X_r = np.array(true_matches).reshape(-1, 1)
    X_w = np.array(false_matches).reshape(-1, 1)
    X = np.concatenate((X_r, X_w))
    Y_r = np.ones(len(true_matches), dtype=int)
    Y_w = np.zeros(len(false_matches), dtype=int)
    Y = np.concatenate((Y_r, Y_w))
    lr = LogisticRegression()
    lr.fit(X, Y)
    f_y = - np.log((1-thresh)/thresh)
    match_thresh = (f_y - lr.intercept_)/lr.coef_
    return match_thresh.item()


def estimate_footprint(
    fm, query_image, surrounding_image, matcher,  surrounding_img_footprint, HW,
    save_images=False, viz_params=None
):
    """
    Parameters
    ----------
    fm : fundamental matrix from previous iteration (None if first iteration).
    query_image : torch.tensor with the query image
    surrounding_image : torch.tensor with the surrounding image
    matcher : a matcher from the image-matching-models.
    surrounding_img_footprint : TODO
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
    TODO

    """
    assert surrounding_image.shape[1] == surrounding_image.shape[2], f"{surrounding_image.shape}"
    
    if fm is not None:
        # Use the FM (i.e. homography) from the previous iteration to generate the new candidate
        transformed_corners = apply_homography_to_corners(HW, HW, fm) + HW
        endpoints = [[HW, HW], [HW*2, HW], [HW*2, HW*2], [HW, HW*2]]
        warped_surrounding_pred_img = tfm.functional.perspective(
            surrounding_image, transformed_corners.numpy(), endpoints, InterpolationMode.BILINEAR
        )
    else:
        warped_surrounding_pred_img = surrounding_image
    
    assert tuple(warped_surrounding_pred_img.shape) == (3, HW*3, HW*3)
    warped_pred_img = warped_surrounding_pred_img[:, HW:HW*2, HW:HW*2]
    
    if save_images:
        tfm.ToPILImage()(warped_pred_img).save(viz_params["pred_path"])
    
    num_inliers, new_fm = compute_matching(
        image0=query_image, image1=warped_pred_img,
        matcher=matcher, save_images=save_images,
        viz_params=viz_params
    )
    
    if num_inliers == 0:
        # If no inliers are found, stop the iterative process
        return num_inliers, None, None, None
    
    if fm is None:  # At the first iteration fm is None
        fm = new_fm
    else:
        fm = add_homographies_fm(fm, new_fm)
    
    transformed_corners = apply_homography_to_corners(HW, HW, fm) + HW

    pred_polygon = shapely.Polygon((transformed_corners - HW) / HW)
    if not pred_polygon.convex_hull.equals(pred_polygon):
        # If the prediction has a non-convex footprint, it is considered not valid
        return -1, None, None, None
    if pred_polygon.area > 9:
        # If the prediction's area is bigger than the surrounding_image's area, it is considered not valid
        return -1, None, None, None

    warped_pred_footprint = apply_homography_to_footprint(surrounding_img_footprint, transformed_corners, HW*3)
    pretty_printed_footprint = "; ".join([f"{lat_lon[0]:.5f}, {lat_lon[1]:.5f}" for lat_lon in warped_pred_footprint])
    return num_inliers, fm, warped_pred_footprint, pretty_printed_footprint


def enlarge_polygon(polygon, scale_factor):
    cntr = polygon.centroid
    scaled_coords = [(cntr.x + (lat - cntr.x) * scale_factor, cntr.y + (lon - cntr.y) * scale_factor)
                     for lat, lon in zip(*polygon.exterior.xy)]
    scaled_polygon = shapely.Polygon(scaled_coords)
    return scaled_polygon


def get_polygon(lats_lons):
    assert isinstance(lats_lons, np.ndarray)
    assert lats_lons.shape == (4, 2)
    polygon = shapely.Polygon(lats_lons)
    return polygon


def get_query_metadata(query_path):
    _, lat, lon, nlat, nlon, tilt, fclt, cldp, mrf, _ = str(query_path).split("@")
    return float(lat), float(lon), int(tilt), int(fclt), int(cldp)


def get_centerpoint_from_query_path(query_path):
    lat, lon, tilt, fclt, cldp = get_query_metadata(query_path)
    return shapely.Point(lat, lon)


def fclt_le_200(query_path):
    lat, lon, tilt, fclt, cldp = get_query_metadata(query_path)
    return fclt <= 200


def fclt_200_400(query_path):
    lat, lon, tilt, fclt, cldp = get_query_metadata(query_path)
    return 200 < fclt <= 400


def fclt_400_800(query_path):
    lat, lon, tilt, fclt, cldp = get_query_metadata(query_path)
    return 400 < fclt <= 800


def fclt_g_800(query_path):
    lat, lon, tilt, fclt, cldp = get_query_metadata(query_path)
    return 800 < fclt


def tilt_l_40(query_path):
    lat, lon, tilt, fclt, cldp = get_query_metadata(query_path)
    return tilt < 40


def tilt_ge_40(query_path):
    lat, lon, tilt, fclt, cldp = get_query_metadata(query_path)
    return tilt >= 40


def cldp_l_40(query_path):
    lat, lon, tilt, fclt, cldp = get_query_metadata(query_path)
    return cldp < 40


def cldp_ge_40(query_path):
    lat, lon, tilt, fclt, cldp = get_query_metadata(query_path)
    return cldp >= 40
