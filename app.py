import cv2
from flask import Flask, request, jsonify, send_file
from PIL import Image
import io
import os

from ocr import *


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Check if image is loaded
        if img is None:
            return jsonify({"error": "Could not decode image"}), 400
 #=============================================================================================       
        resize_ratio = 500 / img.shape[0]

        img = opencv_resize(img, resize_ratio)

        # Convert the image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.GaussianBlur(gray_img, (5,5), 1)

        edged = cv2.Canny(gray_img, 100, 200, apertureSize=3)

        contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # image_with_contours = cv2.drawContours(gray_img, contours, -1, (0,255,0), 3)

        largest_contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
        # image_with_largest_contours = cv2.drawContours(gray_img, largest_contours, -1, (0,255,0), 3)

        receipt_contour = get_receipt_contour(largest_contours)
        image_with_receipt_contour = cv2.drawContours(gray_img, [receipt_contour], -1, (0, 255, 0), 2)

        scanned = wrap_perspective(img.copy(), contour_to_rect(receipt_contour, resize_ratio))
#=============================================================================================
        # Encode the image to a byte stream
        _, buffer = cv2.imencode('.jpg', image_with_receipt_contour)
        byte_io = io.BytesIO(buffer)

        return send_file(byte_io, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
