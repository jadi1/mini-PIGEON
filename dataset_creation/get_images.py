# using the coordinates, try actually downloading the images from google maps!
import requests
from dotenv import load_dotenv
import os

from PIL import Image
from io import BytesIO


load_dotenv()

api_key = os.getenv("MAPS_API_KEY")

def get_image(lat,lng, heading, width, height):
    url = f"https://maps.googleapis.com/maps/api/streetview?size={width}x{height}&location={lat},{lng}&heading={heading}&key={api_key}"
    response = requests.get(url)

    if response.status_code == 200:
        # Convert bytes to image
        img = Image.open(BytesIO(response.content))
        
        img_name = "hey2"
        # Ensure folder exists
        folder = "data/streetview_images"
        os.makedirs(folder, exist_ok=True)

        save_path = f"{folder}/{img_name}.jpg"
        # Save if requested
        if save_path:
            img.save(save_path)
            print(f"Saved image to {save_path}")
        
        # Display the image
        img.show()
        return img
    else:
        print(f"Failed to fetch image. Status code: {response.status_code}")
        return None

# Example usage
img = get_image(-16.793298965782455,-47.61667766559674, heading=90, width=224, height=224)

# helper function to give each image a unique name
# region is GID_1 label
# sample number is nth sample of that region
# pano number is the nth picture of that coordinate
def create_image_name(region, sample_number, pano_number):
    name = f"{region}img{sample_number}no.{pano_number}"
    return name

