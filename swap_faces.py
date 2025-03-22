from flask import Flask, request, jsonify, send_from_directory
from typing import List, Optional, Tuple
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import requests
from io import BytesIO
import base64
from gfpgan import GFPGANer
import subprocess
import traceback
import os
from werkzeug.utils import secure_filename
from google.cloud import storage
import tempfile
import logging
from dataclasses import dataclass

# Ensure the GOOGLE_APPLICATION_CREDENTIALS environment variable is set
# google_credentials = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
# if not google_credentials:
#     raise ValueError("The GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")

# # Initialize Google Cloud Storage Client with the credentials
# client = storage.Client.from_service_account_json(google_credentials)

# Environment variables
BUCKET_NAME = os.environ.get('BUCKET_NAME', 'face-analysis-bucket')

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Function to load model from Google Cloud Storage
MODEL_STORE_DIR = "model_store"  # Define a directory to store models

def load_model_from_gcs(bucket_name, blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Create the model store directory if it does not exist
    if not os.path.exists(MODEL_STORE_DIR):
        os.makedirs(MODEL_STORE_DIR)

    # Construct the path to save the file
    local_path = os.path.join(MODEL_STORE_DIR, blob_name.replace('/', '_'))

    # Only download if the file does not exist
    if not os.path.exists(local_path):
        logging.info(f"Model {blob_name} not found locally. Downloading from GCS...")
        with open(local_path, 'wb') as file_obj:
            blob.download_to_file(file_obj)
        logging.info(f"Downloaded {blob_name} to {local_path}")
    else:
        logging.info(f"Model {blob_name} found locally. Using cached version at {local_path}")

    return local_path

# Flask application initialization
flask_app = Flask(__name__)

bucket_name = 'face-analysis-bucket'

# Load models from Google Cloud Storage
inswapper_model_path = load_model_from_gcs(bucket_name, 'inswapper_128.onnx')
gfpgan_model_path = load_model_from_gcs(bucket_name, 'GFPGANv1.4.pth')

# Initialize InsightFace and GFPGAN
face_analysis_app = FaceAnalysis(name='buffalo_l', root=os.path.dirname(inswapper_model_path))
face_analysis_app.prepare(ctx_id=-1, det_size=(640, 640))
gfpgan = GFPGANer(model_path=gfpgan_model_path, upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=None)

@dataclass
class FaceDetectionResult:
    success: bool
    faces: List[any]
    closest_face: Optional[any]
    error_message: Optional[str] = None


def detect_faces_in_image(img: np.ndarray, face_analysis_app) -> FaceDetectionResult:
    """Helper function to detect faces and find the closest one if multiple exist"""
    try:
        faces = face_analysis_app.get(img)
        
        if len(faces) == 0:
            return FaceDetectionResult(
                success=False,
                faces=[],
                closest_face=None,
                error_message="No face detected"
            )

        if len(faces) == 1:
            return FaceDetectionResult(
                success=True,
                faces=faces,
                closest_face=faces[0]
            )

        # Multiple faces - find the closest (largest) one
        closest_face = max(faces, key=lambda face: 
            (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
        
        return FaceDetectionResult(
            success=True,
            faces=faces,
            closest_face=closest_face
        )

    except Exception as e:
        return FaceDetectionResult(
            success=False,
            faces=[],
            closest_face=None,
            error_message=f"Face detection error: {str(e)}"
        )



# for face swapping
@flask_app.route('/swap-face', methods=['POST'])
def swap_face():
    try:
        # Read source (user's face) and target (generated) images
        source_file = request.files['user_image']
        target_image_url = request.form['generated_image_url']

        # Convert images to OpenCV format
        source_img = cv2.imdecode(np.frombuffer(source_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        
        # Download target image
        response = requests.get(target_image_url)
        target_img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        target_img = cv2.imdecode(target_img_array, cv2.IMREAD_COLOR)

        # Detect faces in both images using our helper function
        source_result = detect_faces_in_image(source_img, face_analysis_app)
        if not source_result.success:
            return jsonify({
                "error": f"Source image: {source_result.error_message}",
                "faces_detected": {"source": len(source_result.faces)}
            }), 400

        target_result = detect_faces_in_image(target_img, face_analysis_app)
        if not target_result.success:
            return jsonify({
                "error": f"Target image: {target_result.error_message}",
                "faces_detected": {"target": len(target_result.faces)}
            }), 400

        # Load face swapper model
        try:
            swapper = insightface.model_zoo.get_model(str(inswapper_model_path), 
                                                    download=False, 
                                                    download_zip=False)
            assert swapper is not None, "Model could not be loaded"
        except Exception as e:
            return jsonify({'error': f"Model loading error: {str(e)}"}), 500

        # Use the closest faces for both images
        swapped_img = swapper.get(
            target_img, 
            target_result.closest_face, 
            source_result.closest_face, 
            paste_back=True
        )

        # Enhance the image using GFPGAN
        _, _, enhanced_img = gfpgan.enhance(
            swapped_img, 
            has_aligned=False, 
            only_center_face=False, 
            paste_back=True
        )

        # Convert to base64
        _, buffer = cv2.imencode('.jpg', enhanced_img)
        res_base64 = base64.b64encode(buffer).decode()

        # Prepare detailed response
        response = {
            'result_image': res_base64,
            'face_detection_details': {
                'source_faces': len(source_result.faces),
                'target_faces': len(target_result.faces),
                'using_closest_faces': len(source_result.faces) > 1 or len(target_result.faces) > 1
            }
        }

        # Add face location info if using closest faces
        if len(source_result.faces) > 1:
            source_bbox = source_result.closest_face.bbox
            response['face_detection_details']['source_face_location'] = {
                "x": int(source_bbox[0]),
                "y": int(source_bbox[1]),
                "width": int(source_bbox[2] - source_bbox[0]),
                "height": int(source_bbox[3] - source_bbox[1])
            }

        if len(target_result.faces) > 1:
            target_bbox = target_result.closest_face.bbox
            response['face_detection_details']['target_face_location'] = {
                "x": int(target_bbox[0]),
                "y": int(target_bbox[1]),
                "width": int(target_bbox[2] - target_bbox[0]),
                "height": int(target_bbox[3] - target_bbox[1])
            }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

    
@flask_app.route('/enhance-face', methods=['POST'])
def enhance_face():
    try:
        # Save uploaded image
        file = request.files['image']
        filename = secure_filename(file.filename)  # Use the secure_filename function to validate the filename
        input_path = os.path.join('inputs/upload', filename)
        file.save(input_path)

        # Read the image
        img = cv2.imread(input_path)

        # Enhance the image using the GFPGAN model
        _, _, enhanced_img = gfpgan.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)

        # Save the enhanced image
        output_path = os.path.join('results', filename)
        cv2.imwrite(output_path, enhanced_img)

        # Encode the image to base64 to return as JSON
        with open(output_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        return jsonify({'enhanced_image': encoded_string})

    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({'error': str(e), 'trace': tb}), 500



if __name__ == '__main__':
    print("Code updated - version")
    # flask_app.run(debug=True, host='0.0.0.0', port=5000)
    flask_app.run(debug=True, host='0.0.0.0', port=5000)