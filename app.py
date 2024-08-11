from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
from ndvi_processor import process_ndvi_change
import google.generativeai as genai
import rasterio
from rasterio.transform import Affine
from geopy.geocoders import Photon
import PIL.Image
from PIL import Image
from dotenv import load_dotenv
import re

load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = 'F:\\hackathon3\\NDVI-Change-Analysis-Tool\\static\\output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables for image paths
imgpath1 = ""
imgpath2 = ""
imgpath3 = ""

api_keys = os.getenv('API_KEY')

# Configure API key for Google Generative AI
genai.configure(api_key=api_keys)

def extract_geotiff_metadata(file_path):
    with rasterio.open(file_path) as src:
        # Extract metadata
        metadata = {
            'CRS': src.crs.to_string(),  # Coordinate Reference System
            'Transform': src.transform,  # Affine transform
            'Bounds': src.bounds,  # Bounding box
            'Width': src.width,  # Image width
            'Height': src.height,  # Image height
            'Number of Bands': src.count,  # Number of bands
            'Dtypes': src.dtypes,  # Data types of bands
            'Tags': src.tags(),  # General tags and metadata
            'Band Metadata': {i: src.descriptions[i] for i in range(src.count)}  # Band-specific descriptions
        }
        
        # Additional information for each band
        band_metadata = {}
        for i in range(1, src.count + 1):
            band = src.read(i)
            band_metadata[f'Band {i}'] = {
                'dtype': band.dtype,
                'shape': band.shape,
                'min': band.min(),
                'max': band.max(),
                'mean': band.mean()
            }
        
        metadata['Band Metadata'] = band_metadata

    return metadata

def pixel_to_geo(x, y, transform):
    lon, lat = transform * (x, y)
    return lon, lat

def get_location_info(latitude, longitude):
    geolocator = Photon(user_agent="measurements")
    location = geolocator.reverse((latitude, longitude), language='en', timeout=5)
    return location.address

def main(file_path, pixel_x, pixel_y):
    # Extract GeoTIFF metadata
    geotiff_metadata = extract_geotiff_metadata(file_path)
    
    # Extract transform matrix from metadata
    transform = geotiff_metadata['Transform']
    
    # Convert pixel coordinates to geographic coordinates
    lon, lat = pixel_to_geo(pixel_x, pixel_y, transform)
    
    # Get location information
    location_info = get_location_info(lat, lon)
    
    # Combine all information into a final dictionary
    final_dict = {
        'GeoTIFF Metadata': geotiff_metadata,
        'Geographic Coordinates': {
            'Longitude': lon,
            'Latitude': lat
        },
        'Location Info': location_info
    }
    
    return final_dict

def convert_tiff_to_png(tiff_path, png_path):
    with rasterio.open(tiff_path) as src:
        # Read the first band
        band = src.read(1)
        # Normalize the band data to 8-bit for PNG conversion
        band = ((band - band.min()) / (band.max() - band.min()) * 255).astype('uint8')
        # Create and save the image using PIL
        img = Image.fromarray(band)
        img.save(png_path, format="PNG")
    # plot_ndvi(ndvi_path_1, [geotransform[0], geotransform[0] + cols * geotransform[1], geotransform[3] + rows * geotransform[5], geotransform[3]], 0, 'YlGn', ndvi_path_1.replace('.tif', '.png'))

def analyze_image_with_model(tiff_image_paths, metadata_dict):
    img_change = PIL.Image.open(tiff_image_paths[0].replace(".tif", ".png"))
    print("imgchange: ", tiff_image_paths[0].replace(".tif", ".png"))
    tiff_image_paths.pop(0)
    print("tifflist: ", tiff_image_paths)
    png_image_paths = [tiff_path.replace('.tif', '.png') for tiff_path in tiff_image_paths]
    
    for tiff_path, png_path in zip(tiff_image_paths, png_image_paths):
        convert_tiff_to_png(tiff_path, png_path)
    
    img_before = PIL.Image.open(png_image_paths[0])
    img_after = PIL.Image.open(png_image_paths[1])
    
    # Initialize the model
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    
    # Create a prompt that includes metadata information
    metadata_str = "\n".join([f"{key}: {value}" for key, value in metadata_dict.items()])
    prompt = (f"Analyse this image thoroughly and it contains the details of NDVI change between two satellite images. Include all your observations and insights about the change detected in terms of NDVI values observed in the image in your response."
              "belonging to two different years. The image passed first is the change in terms of NDVI values. Second and third images are the before and after images of the same area in terms of NDVI values. Here is additional metadata related to the image:\n\n"
              f"{metadata_str}\n\n"
              "Please provide intelligent recommendations regarding efficient land resource usage that can be done to stop or avoid deforestation according to the provided output image (the one depicting the change in NDVI values). NOTE THAT YOU SHOULD PRODUCE A VERY PROFESSIONAL RESPONSE CONTAINING NO SUGGESTIONS OR SPECIAL POINTS WORTH NOTING.")
    
    # Generate content based on the images and metadata
    response = model.generate_content([prompt, img_change, img_before, img_after])
    
    return response.text

def format_response_text(response_text):
    formatted_text = re.sub(r'\*\*|--|##', '', response_text)
    
    # Ensure new lines before numbering
    formatted_text = re.sub(r'(?<!^)\n(\d+\.)', r'<br><br>\1', formatted_text)
    formatted_text = re.sub(r'\n\* ', r'<br><br>* ', formatted_text)
    formatted_text = re.sub(r'\n', '<br>', formatted_text)
    return formatted_text.strip()

def clear_output_folder():
    """Clear the contents of the output folder."""
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

@app.route('/visualize', methods=['GET', 'POST'])
def visualize():
    return render_template("visual.html")

@app.route('/', methods=['GET', 'POST'])
def index():
    global imgpath1, imgpath2, imgpath3
    
    if request.method == 'POST':
        b4_2014 = request.files['b4_2014']
        b5_2014 = request.files['b5_2014']
        b4_2019 = request.files['b4_2019']
        b5_2019 = request.files['b5_2019']
        
        if b4_2014 and b5_2014 and b4_2019 and b5_2019:
            b4_2014_filename = secure_filename(b4_2014.filename)
            b5_2014_filename = secure_filename(b5_2014.filename)
            b4_2019_filename = secure_filename(b4_2019.filename)
            b5_2019_filename = secure_filename(b5_2019.filename)
            
            b4_2014_year = extract_year(b4_2014_filename)
            b5_2014_year = extract_year(b5_2014_filename)
            b4_2019_year = extract_year(b4_2019_filename)
            b5_2019_year = extract_year(b5_2019_filename)
            
            b4_2014_path = os.path.join(app.config['UPLOAD_FOLDER'], f'B4_{b4_2014_year}.tif')
            b5_2014_path = os.path.join(app.config['UPLOAD_FOLDER'], f'B5_{b5_2014_year}.tif')
            b4_2019_path = os.path.join(app.config['UPLOAD_FOLDER'], f'B4_{b4_2019_year}.tif')
            b5_2019_path = os.path.join(app.config['UPLOAD_FOLDER'], f'B5_{b5_2019_year}.tif')
            
            b4_2014.save(b4_2014_path)
            b5_2014.save(b5_2014_path)
            b4_2019.save(b4_2019_path)
            b5_2019.save(b5_2019_path)
            
            ndvi_2014_path = os.path.join(app.config['UPLOAD_FOLDER'], f'NDVI_{b4_2014_year}.tif')
            ndvi_2019_path = os.path.join(app.config['UPLOAD_FOLDER'], f'NDVI_{b4_2019_year}.tif')
            ndvi_change_path = os.path.join(app.config['UPLOAD_FOLDER'], 'NDVIChange.tif')
            ndvi_change_image = os.path.join(app.config['UPLOAD_FOLDER'], 'NDVIChange.png')
                
            imgpath1 = ndvi_change_path
            imgpath2 = ndvi_2014_path
            imgpath3 = ndvi_2019_path
            process_ndvi_change(b4_2014_path, b5_2014_path, b4_2019_path, b5_2019_path, ndvi_2014_path, ndvi_2019_path, ndvi_change_path, ndvi_change_image)

            ndvi_2014_png = url_for('static', filename=f'output/NDVI_{b4_2014_year}.png')
            ndvi_2019_png = url_for('static', filename=f'output/NDVI_{b5_2019_year}.png')
            return render_template('index.html', ndvi_2014_png=ndvi_2014_png, ndvi_2019_png=ndvi_2019_png, 
                                   ndvi_2014_year=b4_2014_year, ndvi_2019_year=b4_2019_year)
    
    return render_template('index.html')

@app.route('/result')
def result():
    global imgpath1, imgpath2, imgpath3
    
    ndvi_change_png = url_for('static', filename='output/NDVIChange.png')
    pixel_x = 1000
    pixel_y = 1000
    tiff_image_paths = [imgpath1, imgpath2, imgpath3]
    print("tiff img: ", tiff_image_paths)
    metadata_dict = main(tiff_image_paths[0], pixel_x, pixel_y)
    response_text = analyze_image_with_model(tiff_image_paths, metadata_dict)
    formatted_text = format_response_text(response_text)
    print(response_text)
    print(ndvi_change_png)
    return render_template('result.html', ndvi_change_png=ndvi_change_png, response_text=formatted_text)

@app.route('/clear', methods=['POST'])
def clear_files():
    """Route to handle clearing all files in the output folder."""
    clear_output_folder()
    return redirect(url_for('index'))

def extract_year(filename):
    """Extracts the year from the filename in the format 'YYYY'."""
    # Split filename by underscores to isolate date sections
    parts = filename.split('_')
    
    # Look for sections that start with 'YYYY'
    for part in parts:
        if len(part) >= 4 and part[:4].isdigit():
            year = part[:4]  # Extract the first four digits
            return year
    return None

if __name__ == '__main__':
    app.run(debug=True)