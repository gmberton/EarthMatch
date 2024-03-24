# EarthMatch

## Download the data

Simply run 
```
python data_downloader.py
```
which will download the data in the right format for the benchmark.

## Run experiment

You can run
```
python main.py --matcher sift-lg --max_num_keypoints 2048 --img_size 512 --save_images
```
which will run the experiment and save all visualizations in `./logs/`
