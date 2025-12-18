# using the coordinates, try actually downloading the images from google maps!
import requests
from dotenv import load_dotenv
import os
import pandas as pd
import random

from PIL import Image
from io import BytesIO

load_dotenv()

# ensure folder exists
os.makedirs("data/streetview_images", exist_ok=True)

def get_image(img_name, lat,lng, heading, width, height, api_key, folder):
    url = f"https://maps.googleapis.com/maps/api/streetview?size={width}x{height}&location={lat},{lng}&heading={heading}&key={api_key}"
    response = requests.get(url)

    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        
        save_path = os.path.join(folder, f"{img_name}.jpg")
        if save_path:
            img.save(save_path)
            print(f"Saved image to {save_path}")
        
        return save_path
    else:
        print(f"Failed to fetch image. Status code: {response.status_code}")
        return None
    
# generates 4 images, returns 4 filepaths to the images
def generate_panorama(region, lat, lng, api_key, folder):
    imgs = []
    heading = random_heading() # generate random heading
    for j in range(0,4):
        img_name = create_image_name(region,lat, lng,j+1)
        img_path = get_image(img_name, lat,lng, heading, 500, 500, api_key, folder) # 500 by 500 images
        imgs.append(img_path)
        heading = heading + 90 # update heading

    # return the 4 paths
    return imgs

# helper function to give each image a unique name
# region is GID_1 label
# sample number is nth sample of that region
# pano number is the nth picture of that coordinate
def create_image_name(region, lat, lng, pano_number):
    lat_str = f"{lat:.6f}".replace('.', '_')  # 6 decimal places, replace '.' with '_'
    lng_str = f"{lng:.6f}".replace('.', '_')
    name = f"{region}_{lat_str}_{lng_str}.{pano_number}"
    return name

# helper function to generate a random direction from 0 to 90 degrees
def random_heading():
    return random.randint(0, 89)


# read the coordinates csv and generate 4 images per coordinate
def generate_images_from_coords(dataset_csv, start_idx, end_idx, api_key=os.getenv("MAPS_API_KEY")):
    # read csv
    coordinates = pd.read_csv("data/coordinates_data_final.csv")
    
    # load existing dataset csv if it exists
    if os.path.exists(dataset_csv):
        dataset_df = pd.read_csv(dataset_csv)
    else:
        dataset_df = pd.DataFrame(columns=["region", "lat", "lng"])
    
    # keep track of which coordinates have already been processed
    processed = set(dataset_df[["region", "lat", "lng"]].itertuples(index=False, name=None))

    for i in range(start_idx, end_idx):
        row = coordinates.iloc[i]
        region = row["ADM1_Region"]
        lat = row["Latitude"]
        lng = row["Longitude"]

        # skip if already processed
        if (region, lat, lng) in processed:
            continue

        # check if folder for this region exists, if not, make it
        folder = f"data/streetview_images/{region}"
        os.makedirs(folder, exist_ok=True)

        # for each image, get 4 images in panorama
        img_paths = generate_panorama(region, lat, lng, api_key, folder = folder)

        dataset_df = pd.concat([dataset_df, pd.DataFrame([{
            "region": region,
            "lat": lat,
            "lng": lng,
            "img1": img_paths[0],
            "img2": img_paths[1],
            "img3": img_paths[2],
            "img4": img_paths[3],
        }])], ignore_index=True)

        # save to csv
        dataset_df.to_csv(dataset_csv, index=False)

generate_images_from_coords("data/img_dataset.csv", 10890, 12743) 