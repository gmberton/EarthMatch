
import cv2
import sys
import torch
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from collections import defaultdict
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

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--matcher", type=str, default="sift-lg", help="_")
parser.add_argument("-nk", "--max_num_keypoints", type=int, default=2048, help="_")
parser.add_argument("-ni", "--num_iterations", type=int, default=4, help="_")
parser.add_argument("-is", "--img_size", type=int, default=1024, help="_")
parser.add_argument("--save_images", action='store_true', help="_")
parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="_")
parser.add_argument("--data_dir", type=str, default="./data", help="_")
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

matcher = get_matcher(args.matcher, device=args.device, max_num_keypoints=args.max_num_keypoints)

queries_input_folders = sorted(list(args.data_dir.glob("*")))

all_results = []

for folder in tqdm(queries_input_folders):
    paths = sorted(list(folder.glob("*")))
    assert len(paths) == 11  # One query and its 10 predictions, therefore 11 files
    query_path = paths[0]
    query_centerpoint = util_matching.get_centerpoint_from_query_path(query_path)
    preds_paths = paths[1:]
    for pred_idx, surrounding_pred_path in enumerate(preds_paths):
        try:
            query_log_dir = log_dir / query_path.stem / f"{pred_idx:02d}"
            rot_angle = int(surrounding_pred_path.name.split("__")[2].replace("rot", ""))
            assert rot_angle % 90 == 0
            query_image = matcher.image_loader(query_path, args.img_size).to(args.device)
            query_image = tfm.functional.rotate(query_image, rot_angle)
            surrounding_image = matcher.image_loader(surrounding_pred_path, args.img_size*3).to(args.device)
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
                pred_polygon = util_matching.get_polygon(predicted_footprint.numpy())
                pred_polygon = util_matching.enlarge_polygon(pred_polygon, 3)
                if pred_polygon.contains(query_centerpoint):
                    logging.debug(f"{query_path.stem} {pred_idx=} {iteration=:02d} MSG2_FOUND_CORRECT {num_inliers=} pred={pretty_printed_footprint}")
                    if iteration == args.num_iterations - 1:
                        all_results.append((query_path.stem, pred_idx, num_inliers, predicted_footprint, True))
                else:
                    logging.debug(f"{query_path.stem} {pred_idx=} {iteration=:02d} MSG3_FOUND_WRONG {num_inliers=} pred={pretty_printed_footprint}")
                    if iteration == args.num_iterations - 1:
                        all_results.append((query_path.stem, pred_idx, num_inliers, predicted_footprint, False))
            
        except (ValueError, torch._C._LinAlgError, cv2.error, IndexError, AttributeError) as e:
            logging.debug(f"{query_path.stem} {pred_idx=} {iteration=:02d} MSG4_ERROR Error {e}")

torch.save(all_results, log_dir / "results.torch")

num_inliers_for_true_positives = [res[2] for res in all_results if res[-1] is True]
num_inliers_for_false_positives = [res[2] for res in all_results if res[-1] is False]
if len(num_inliers_for_false_positives) == 0:
    # The model never reached convergence for a false positive
    threshold = -1
else:
    threshold = util_matching.compute_threshold(num_inliers_for_true_positives, num_inliers_for_false_positives, thresh=0.999)

results_per_query = defaultdict(list)
for res in all_results:
    results_per_query[res[0]].append(res)

located = 0
located_fclt_le_200 = 0
located_fclt_200_400 = 0
located_fclt_400_800 = 0
located_fclt_g_800 = 0
located_tilt_ge_40 = 0
located_tilt_l_40 = 0
located_cldp_ge_40 = 0
located_cldp_l_40 = 0
for query_name, results in results_per_query.items():
    for _, _, num_inliers, _, is_correct in results:
        if num_inliers >= threshold:
            located += 1
            if util_matching.fclt_le_200(query_name):
                located_fclt_le_200 += 1
            if util_matching.fclt_200_400(query_name):
                located_fclt_200_400 += 1
            if util_matching.fclt_400_800(query_name):
                located_fclt_400_800 += 1
            if util_matching.fclt_g_800(query_name):
                located_fclt_g_800 += 1
            if util_matching.tilt_ge_40(query_name):
                located_tilt_ge_40 += 1
            if util_matching.tilt_l_40(query_name):
                located_tilt_l_40 += 1
            if util_matching.cldp_ge_40(query_name):
                located_cldp_ge_40 += 1
            if util_matching.cldp_l_40(query_name):
                located_cldp_l_40 += 1
            break

logging.info(f"{threshold=}")
logging.info(
    f"{located=} "
    f"{located_fclt_le_200=} {located_fclt_200_400=} {located_fclt_400_800=} {located_fclt_g_800=} "
    f"{located_tilt_l_40=} {located_tilt_ge_40=} "
    f"{located_cldp_l_40=} {located_cldp_ge_40=}"
)
