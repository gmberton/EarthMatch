# EarthMatch

EarthMatch is an image-matching / coregistration pipeline used to localize photos taken by astronauts aboard the ISS. It takes as input a pair of images, the astronaut photo to be localized and a potential candidate (obtain from a retrieval method like EarthLoc) and, if the two images do overlap, it outputs their precise coregistration.

[Check out our webpage](https://earthloc-and-earthmatch.github.io/)

The paper, called "EarthMatch: Iterative Coregistration for Fine-grained Localization of Astronaut Photography" is accepted to the 2024 CVPR workshop of "Image Matching: Local Features & Beyond 2024".

## Run the experiments

```
# Clone the repo
git clone --recursive https://github.com/gmberton/EarthMatch
cd EarthMatch
# Download the data
rsync -rhz --info=progress2 --ignore-existing rsync://vandaldata.polito.it/sf_xl/EarthMatch/data .
# Run the experiment with SIFT-LightGlue
python main.py --matcher sift-lg --max_num_keypoints 2048 --img_size 512 --data_dir data --log_dir out_sift-lg --save_images
```

The data contains 268 astronaut photos and, for each of them, the top-10 predictions obtained from a worldwide database with an enhanced version of EarthMatch.

The logs and visualizations will be automatically saved in `./logs/out_sift-lg` (note that using `--save_images` will save images for all results and slow down the experiment.

You can set the matcher to any of the 17 matchers used in the [image-matching-models repo](https://github.com/gmberton/image-matching-models).


## Cite
```
@InProceedings{Berton_2024_EarthMatch,
    author    = {Gabriele Berton, Gabriele Goletto, Gabriele Trivigno, Alex Stoken, Barbara Caputo, Carlo Masone},
    title     = {EarthMatch: Iterative Coregistration for Fine-grained Localization of Astronaut Photography},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
}
```
