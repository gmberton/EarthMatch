
import io
import time
import random
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from urllib.request import Request, urlopen


YEAR = 2018

DATABASE_URL = "https://s2maps-tiles.eu/wmts?layer=s2cloudless-__YEAR___3857&" \
    "style=default&tilematrixset=GoogleMapsCompatible&Service=WMTS" \
    "&Request=GetTile&Version=1.0.0&Format=image%2Fjpeg" \
    "&TileMatrix=__ZOOM__&TileCol=__COL__&TileRow=__ROW__"

QUERIES_URL = "https://eol.jsc.nasa.gov/DatabaseImages/ESC/large"


def image_save_atomically(pil_img, dst_img_path):
    """Save image in an atomic procedure, so that interrupting the process does
    not leave a corrupt image saved, so that image download can be interrupted
    safely at any moment."""
    dst_img_path.parent.mkdir(parents=True, exist_ok=True)
    # A hidden file to temporarily store the image during its creation
    tmp_img_path = "." + str(random.randint(0, 99999999999)) + dst_img_path.suffix
    pil_img.save(tmp_img_path)
    shutil.move(tmp_img_path, dst_img_path)


def download_image_from_url(url, num_tries=4):
    """Return RGB PIL Image from its URL"""
    for i in range(num_tries):  # Try the download num_tries
        try:
            req = Request(url=url, headers={'User-Agent': 'Mozilla/5.0'})
            req = urlopen(req, timeout=5)
            return Image.open(io.BytesIO(req.read())).convert("RGB")
        except Exception as e:
            if i == num_tries:
                raise e
            else:
                print(f"Couldn't download {url} due to {e}, will retry in 10 seconds")
                time.sleep(10)


with open("images_paths.txt") as file:
    lines = file.read().splitlines()

with open("aims_data.csv") as file:
    aims_data = file.read().splitlines()[1:]
    # data = fclt, cldp, lat, lon, nlat, nlon, tilt
    d_mrf_data = {l.split(",")[0]: l.split(",")[1:] for l in lines}

data_dir = Path("./data")

for i in tqdm(range(0, len(lines), 11), desc="Downloading images, this is slow..."):
    # MRF is mission-roll-frame
    mrf, query_filename = lines[i].split("/")
    mission = mrf.split("-")[0]
    
    q_path = data_dir / mrf / query_filename
    
    if not q_path.exists():
        q_url = f"{QUERIES_URL}/{mission}/{mrf}.JPG"
        q_img = download_image_from_url(q_url)
        image_save_atomically(q_img, q_path)
    
    candidates_paths = [l for l in lines[i+1 : i+11]]
    for candidate_path in candidates_paths:
        # MRF is the MRF of the query
        mrf, candidate_filename = candidate_path.split("/")
        zoom_row_col = candidate_filename.split("@")[9]
        # Zoom, row, col and year are identifiers of the map tile
        zoom, row, col = [int(e) for e in zoom_row_col.split("_")]
        
        image_path = data_dir / mrf / candidate_filename
        
        if image_path.exists():
            continue
        
        tile_mosaic = Image.new("RGB", (256*12, 256*12))
        image = np.zeros([256*12, 256*12, 3], dtype=np.uint8)
        for r in range(12):
            for c in range(12):
                if row-4+r >= 2**zoom or col-4+c >= 2**zoom:
                    continue
                url = DATABASE_URL\
                    .replace('__YEAR__', str(YEAR)).replace('__ZOOM__', str(zoom))\
                    .replace('__ROW__', str(row-4+r)).replace('__COL__', str(col-4+c))
                tile = download_image_from_url(url)
                left = c * 256
                upper = r * 256
                tile_mosaic.paste(tile, (left, upper))
        
        image_save_atomically(tile_mosaic, image_path)

