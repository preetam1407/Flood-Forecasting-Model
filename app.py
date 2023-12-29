from flask import Flask, send_file, request, jsonify, send_file
from flask_cors import CORS
# from thresholding_model_class import ThresholdingModel, GroundTruthMeasurement
from trained_model import pre_trained_model
from cloudinary import config, uploader
from cloudinary.uploader import upload
import os
import rasterio
from rasterio import Affine, MemoryFile
from dotenv import load_dotenv



load_dotenv()

app = Flask(__name__, static_folder='./build', static_url_path='/')
CORS(app)   

config(
    cloud_name=os.getenv('CLOUD_NAME'),
    api_key=os.getenv('API_KEY'),
    api_secret=os.getenv('API_SECRET')
)


@app.route('/', defaults={'path': ''})
@app.route('/<path>')
def index(path):
    return app.send_static_file('index.html')



@app.route('/api/send_measurement', methods=['POST'])
def receive_measurement():
    data = request.get_json()
    measurement = data.get('measurement')

    # try:
        # predicted_result=flood_forecasting_model(measurement)
    predicted_result=pre_trained_model(measurement)

    with rasterio.open("sample_tif.tif") as src:
        transform = src.transform
        width = src.width
        height = src.height

    # Create a new in-memory GeoTIFF file for the prediction
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            width=width,
            height=height,
            count=1,  # Number of bands
            dtype='uint8',
            crs=src.crs,
            #crs="EPSG:4326",
            transform=transform,
        ) as dst:
            # Write the predicted result to the GeoTIFF
            dst.write(predicted_result, 1)

        uploaded_file = upload(memfile, resource_type="raw", format="tif", public_id="predicted_inundation_map")
        file_link = uploaded_file["secure_url"]
        print("Result Sent: ",file_link)
        
    response = {
        "message": "Model Run successfully",
        "file_link": file_link,
    }
    
    return jsonify(response)    
    # except Exception as e:
    #     return jsonify({"message":"An error occurred",})





if __name__ == '__main__':
    app.run(debug=True)
