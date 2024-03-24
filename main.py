
import cv2
import sys
import torch
import logging
import shapely
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import torchvision.transforms as tfm

sys.path.append(str(Path('image-matching-models')))
sys.path.append(str(Path('image-matching-models/third_party/RoMa')))
sys.path.append(str(Path('image-matching-models/third_party/duster')))
sys.path.append(str(Path('image-matching-models/third_party/DeDoDe')))
sys.path.append(str(Path('image-matching-models/third_party/Steerers')))
sys.path.append(str(Path('image-matching-models/third_party/Se2_LoFTR')))
sys.path.append(str(Path('image-matching-models/third_party/LightGlue')))
sys.path.append(str(Path('image-matching-models/third_party/imatch-toolbox')))

import commons
import util_matching
from matching import get_matcher

torch.set_grad_enabled(False)


def get_polygon(lats_lons):
    assert isinstance(lats_lons, np.ndarray)
    assert lats_lons.shape == (4, 2)
    polygon = shapely.Polygon(lats_lons)
    return polygon


def get_centerpoint_from_query_path(query_path):
    center_lat, center_lon = query_path.name.split("@")[1:3]
    return shapely.Point(float(center_lat), float(center_lon))


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--method", type=str, default="sift-lg", help="_")
parser.add_argument("-nk", "--max_num_keypoints", type=int, default=2048, help="_")
parser.add_argument("-ni", "--num_iterations", type=int, default=10, help="_")
parser.add_argument("-is", "--img_size", type=int, default=1024, help="_")
parser.add_argument("--save_images", action='store_true', help="_")
parser.add_argument("--data_dir", type=str, default="./data", help="_")
parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="_")
parser.add_argument("--log_dir", type=str, default="default",
                    help="name of directory on which to save the logs, under logs/log_dir")

args = parser.parse_args()
start_time = datetime.now()
log_dir = Path("logs") / args.log_dir / start_time.strftime('%Y-%m-%d_%H-%M-%S')
commons.setup_logging(log_dir, stdout="info")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {log_dir}")

args.data_dir = Path(args.data_dir)
assert args.data_dir.exists()

matcher = get_matcher(args.method, device=args.device, max_num_keypoints=args.max_num_keypoints)

MRFS = ['ISS050-E-35441', 'ISS050-E-70536', 'ISS051-E-12869', 'ISS051-E-43872', 'ISS051-E-52457', 'ISS052-E-10192', 'ISS052-E-14188', 'ISS052-E-27335', 'ISS052-E-39552', 'ISS052-E-43456', 'ISS052-E-44668', 'ISS052-E-46802', 'ISS052-E-78671', 'ISS052-E-78678', 'ISS052-E-8329', 'ISS052-E-85735', 'ISS053-E-127015', 'ISS053-E-127680', 'ISS053-E-155926', 'ISS054-E-21023', 'ISS055-E-109234', 'ISS055-E-115021', 'ISS055-E-115078', 'ISS055-E-27330', 'ISS055-E-27340', 'ISS055-E-69930', 'ISS055-E-7189', 'ISS055-E-7427', 'ISS055-E-7431', 'ISS056-E-10031', 'ISS056-E-14210', 'ISS056-E-14619', 'ISS056-E-14675', 'ISS056-E-14696', 'ISS056-E-14729', 'ISS056-E-153176', 'ISS056-E-153349', 'ISS056-E-153544', 'ISS056-E-153726', 'ISS056-E-153733', 'ISS056-E-153798', 'ISS056-E-157171', 'ISS056-E-157239', 'ISS056-E-157303', 'ISS056-E-158784', 'ISS056-E-181034', 'ISS056-E-181198', 'ISS056-E-21245', 'ISS056-E-2765', 'ISS056-E-32541', 'ISS056-E-32663', 'ISS056-E-6047', 'ISS056-E-6052', 'ISS056-E-6274', 'ISS056-E-6313', 'ISS057-E-51393', 'ISS059-E-104575', 'ISS059-E-104582', 'ISS059-E-104735', 'ISS059-E-114423', 'ISS059-E-121551', 'ISS059-E-27749', 'ISS059-E-34616', 'ISS059-E-38351', 'ISS060-E-55269', 'ISS060-E-55334', 'ISS060-E-8863', 'ISS061-E-112026', 'ISS061-E-112778', 'ISS061-E-113051', 'ISS061-E-123201', 'ISS061-E-139004', 'ISS061-E-148981', 'ISS061-E-151890', 'ISS061-E-21927', 'ISS061-E-25978', 'ISS061-E-3832', 'ISS061-E-52913', 'ISS061-E-63091', 'ISS061-E-63295', 'ISS062-E-103378', 'ISS062-E-103381', 'ISS062-E-116618', 'ISS062-E-117992', 'ISS062-E-118049', 'ISS062-E-118221', 'ISS062-E-121151', 'ISS062-E-124642', 'ISS062-E-124734', 'ISS062-E-125270', 'ISS062-E-136816', 'ISS062-E-138041', 'ISS062-E-139890', 'ISS062-E-140007', 'ISS062-E-140037', 'ISS062-E-140118', 'ISS062-E-140125', 'ISS062-E-140164', 'ISS062-E-140874', 'ISS062-E-140991', 'ISS062-E-142001', 'ISS062-E-142182', 'ISS062-E-142670', 'ISS062-E-148242', 'ISS062-E-148876', 'ISS062-E-148896', 'ISS062-E-148931', 'ISS062-E-149132', 'ISS062-E-149138', 'ISS062-E-149143', 'ISS062-E-149363', 'ISS062-E-149456', 'ISS062-E-149461', 'ISS062-E-149481', 'ISS062-E-149756', 'ISS062-E-149778', 'ISS062-E-149793', 'ISS062-E-150373', 'ISS062-E-150396', 'ISS062-E-150474', 'ISS062-E-150758', 'ISS062-E-150762', 'ISS062-E-151319', 'ISS062-E-151424', 'ISS062-E-151460', 'ISS062-E-151502', 'ISS062-E-1928', 'ISS062-E-24032', 'ISS062-E-44331', 'ISS062-E-44427', 'ISS062-E-44701', 'ISS062-E-44729', 'ISS062-E-44740', 'ISS062-E-45083', 'ISS062-E-50961', 'ISS062-E-51147', 'ISS062-E-51428', 'ISS062-E-51456', 'ISS062-E-51501', 'ISS062-E-53376', 'ISS062-E-53422', 'ISS062-E-53508', 'ISS062-E-53554', 'ISS062-E-53562', 'ISS062-E-55144', 'ISS062-E-55264', 'ISS062-E-55356', 'ISS062-E-55454', 'ISS062-E-55467', 'ISS062-E-55475', 'ISS062-E-55492', 'ISS062-E-55515', 'ISS062-E-71256', 'ISS062-E-71288', 'ISS062-E-71301', 'ISS062-E-75466', 'ISS062-E-75554', 'ISS062-E-75569', 'ISS062-E-75583', 'ISS062-E-78936', 'ISS062-E-78968', 'ISS062-E-78994', 'ISS062-E-79034', 'ISS062-E-79079', 'ISS062-E-79105', 'ISS062-E-81194', 'ISS062-E-81292', 'ISS062-E-81358', 'ISS062-E-81917', 'ISS062-E-81979', 'ISS062-E-82022', 'ISS062-E-82031', 'ISS062-E-85080', 'ISS062-E-87475', 'ISS062-E-87848', 'ISS062-E-87862', 'ISS062-E-96492', 'ISS062-E-96495', 'ISS062-E-98517', 'ISS063-E-16668', 'ISS063-E-16723', 'ISS063-E-16791', 'ISS063-E-22174', 'ISS063-E-2629', 'ISS063-E-26951', 'ISS063-E-27062', 'ISS063-E-34454', 'ISS063-E-357', 'ISS063-E-38377', 'ISS063-E-38590', 'ISS063-E-398', 'ISS063-E-406', 'ISS063-E-408', 'ISS063-E-60809', 'ISS063-E-60817', 'ISS063-E-631', 'ISS063-E-76868', 'ISS063-E-77146', 'ISS064-E-10105', 'ISS064-E-10117', 'ISS064-E-10232', 'ISS064-E-10235', 'ISS064-E-10238', 'ISS064-E-10325', 'ISS064-E-10568', 'ISS064-E-10999', 'ISS064-E-11005', 'ISS064-E-11006', 'ISS064-E-11477', 'ISS064-E-14463', 'ISS064-E-15697', 'ISS064-E-16203', 'ISS064-E-16221', 'ISS064-E-20444', 'ISS064-E-20457', 'ISS064-E-21388', 'ISS064-E-23288', 'ISS064-E-23290', 'ISS064-E-23317', 'ISS064-E-2825', 'ISS064-E-351', 'ISS064-E-358', 'ISS064-E-35934', 'ISS064-E-37348', 'ISS064-E-38911', 'ISS064-E-40713', 'ISS064-E-43529', 'ISS064-E-4440', 'ISS064-E-44461', 'ISS064-E-44696', 'ISS064-E-46242', 'ISS064-E-47991', 'ISS064-E-48215', 'ISS064-E-48614', 'ISS064-E-48617', 'ISS064-E-48630', 'ISS064-E-49135', 'ISS064-E-49549', 'ISS064-E-49595', 'ISS064-E-49723', 'ISS064-E-513', 'ISS064-E-51351', 'ISS064-E-554', 'ISS064-E-561', 'ISS064-E-56550', 'ISS064-E-56557', 'ISS064-E-56797', 'ISS064-E-56981', 'ISS064-E-56990', 'ISS064-E-57009', 'ISS064-E-57500', 'ISS064-E-58334', 'ISS064-E-58655', 'ISS064-E-58656', 'ISS064-E-58697', 'ISS064-E-58991', 'ISS064-E-59033', 'ISS064-E-59170', 'ISS064-E-59866', 'ISS064-E-60181', 'ISS064-E-60466', 'ISS064-E-60510', 'ISS064-E-60651', 'ISS064-E-60654', 'ISS064-E-6227', 'ISS064-E-7119', 'ISS064-E-7456', 'ISS064-E-7473', 'ISS064-E-7487', 'ISS064-E-7489', 'ISS064-E-7547', 'ISS064-E-7575', 'ISS064-E-7760', 'ISS064-E-7776', 'ISS064-E-7778', 'ISS064-E-8732', 'ISS064-E-9007', 'ISS064-E-9075', 'ISS064-E-9097', 'ISS064-E-9417', 'ISS064-E-9755', 'ISS064-E-9781', 'ISS065-E-111745', 'ISS065-E-1274', 'ISS065-E-133229', 'ISS065-E-143021', 'ISS065-E-290654', 'ISS065-E-31111', 'ISS065-E-389699', 'ISS065-E-389715', 'ISS065-E-86167', 'ISS065-E-93057', 'ISS066-E-125223', 'ISS066-E-134231', 'ISS066-E-134291', 'ISS066-E-134300', 'ISS066-E-138984', 'ISS066-E-156282', 'ISS066-E-84247', 'ISS067-E-1286', 'ISS067-E-133364', 'ISS067-E-148092', 'ISS067-E-174193', 'ISS067-E-174239', 'ISS067-E-315691', 'ISS067-E-35805', 'ISS067-E-7807']
preds_folders = [args.data_dir / mrf for mrf in MRFS]
for folder in preds_folders:
    assert folder.exists()

for folder in tqdm(preds_folders, ncols=120):
    paths = sorted(glob(folder + "/*"))
    assert len(paths) == 11
    query_path = Path(paths[0])
    query_centerpoint = get_centerpoint_from_query_path(query_path)
    preds_paths = paths[1:]
    for pred_idx, surrounding_pred_path in enumerate(preds_paths):
        try:
            query_log_dir = log_dir / query_path.stem / f"{pred_idx:02d}"
            surrounding_pred_path = Path(surrounding_pred_path)
            rot_angle = int(surrounding_pred_path.name.split("__")[2].replace("rot", ""))
            assert rot_angle % 90 == 0
            query_image = matcher.image_loader(query_path, args.img_size, args.device)
            query_image = tfm.functional.rotate(query_image, rot_angle)
            surrounding_image = matcher.image_loader(surrounding_pred_path, args.img_size*3, args.device)
            surrounding_img_footprint = util_matching.path_to_footprint(surrounding_pred_path)
            
            if args.save_images:
                query_log_dir.mkdir(exist_ok=True, parents=True)
                tfm.ToPILImage()(query_image).save(query_log_dir / query_path.name)
                tfm.ToPILImage()(surrounding_image).save(query_log_dir / "surrounding_img.jpg")
            
            fm = None
            found_match = True
            for iteration in range(args.num_iterations):
                viz_params = {
                    "output_dir": query_log_dir,
                    "output_file_suffix": iteration,
                    "query_path": query_path,
                    "pred_path": query_log_dir / f"pred_{iteration}.jpg",
                }
                num_inliers, fm, predicted_footprint, pretty_printed_footprint = util_matching.estimate_footprint(
                    fm,
                    query_image,
                    surrounding_image,
                    matcher,
                    surrounding_img_footprint,
                    HW=args.img_size,
                    save_images=args.save_images,
                    viz_params=viz_params
                )
                if num_inliers == 0:
                    found_match = False
                    logging.debug(f"{query_path.stem} {pred_idx=} {iteration=:02d} MSG1_NOT_FOUND {num_inliers=}")
                    break
                pred_polygon = get_polygon(predicted_footprint.numpy()).convex_hull  # TODO probably convex_hull can be removed
                if pred_polygon.contains(query_centerpoint):
                    logging.debug(f"{query_path.stem} {pred_idx=} {iteration=:02d} MSG2_FOUND_CORRECT {num_inliers=} pred={pretty_printed_footprint}")
                else:
                    logging.debug(f"{query_path.stem} {pred_idx=} {iteration=:02d} MSG3_FOUND_WRONG {num_inliers=} pred={pretty_printed_footprint}")
            
        except (ValueError, torch._C._LinAlgError, cv2.error, IndexError) as e:
            logging.info(f"{query_path.stem} {pred_idx=} {iteration=:02d} MSG4_ERROR Error {e}")
