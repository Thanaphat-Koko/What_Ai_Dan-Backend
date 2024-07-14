import cv2
from flask import Flask, request, jsonify, send_file
from PIL import Image
import io
import os

from ocr import *

import pytesseract
import re

from pytesseract import Output
import easyocr
import matplotlib.pyplot as plt

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
        original = img.copy()
        resize_ratio = 500 / img.shape[0]
        
        # print(img.shape[0])

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
        image_with_receipt_contour = cv2.drawContours(img.copy(), [receipt_contour], -1, (0, 255, 0), 2)

        print(contour_to_rect(receipt_contour, resize_ratio))
        scanned = wrap_perspective(original.copy(), contour_to_rect(receipt_contour, resize_ratio))

        n_gray = cv2.cvtColor(scanned, cv2.COLOR_BGR2GRAY)
        n_gray = cv2.GaussianBlur(n_gray, (5,5), 1)

        High_val_img = n_gray[60:190, 100:372]
        Low_val_img = n_gray[198:323, 100:372]
        Pulse_val_img = n_gray[335:435, 100:372]

        thres, res = cv2.threshold(High_val_img, 50 ,255,cv2.THRESH_BINARY_INV)
        thres2, res2 = cv2.threshold(Low_val_img, 50 ,255,cv2.THRESH_BINARY_INV)
        thres3, res3 = cv2.threshold(Pulse_val_img, 50 ,255,cv2.THRESH_BINARY_INV)

        kernel = np.ones((5,5), np.uint8)
        img_dilation3 = cv2.dilate(res3, kernel, iterations=2)

        d = pytesseract.image_to_data(img_dilation3, output_type=Output.DICT)
        n_boxes = len(d['level'])
        boxes = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        for i in range(n_boxes):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            boxes = cv2.rectangle(boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

        extracted_text = pytesseract.image_to_string(img_dilation3)
        print(extracted_text)
        # output = Image.fromarray(boxes)

        # output = Image.fromarray(img_dilation3)
        # output.save('result.jpg')

        # reader = easyocr.Reader(['en'], gpu = False)
        # results = reader.readtext('result.jpg')
        # for (bbox, text, prob) in results:
        #     print(text)

#=============================================================================================
        # output = Image.fromarray(scanned)
        # output.save('result.png')

        # Encode the image to a byte stream
        _, buffer = cv2.imencode('.jpg', boxs)
        byte_io = io.BytesIO(buffer)

        return send_file(byte_io, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
