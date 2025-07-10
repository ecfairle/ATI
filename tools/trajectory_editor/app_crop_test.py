# Minimal Flask app just for testing the crop feature
from flask import Flask, request, jsonify, render_template, send_file
import os
import io
import base64
from PIL import Image

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/crop')
def crop():
    return render_template('crop.html')

@app.route('/crop_image', methods=['POST'])
def crop_image():
    data = request.get_json()
    
    # Extract crop parameters
    x = data['x']
    y = data['y']
    width = data['width']
    height = data['height']
    target_width = data['targetWidth']
    target_height = data['targetHeight']
    image_data = data['imageData']
    
    # Decode base64 image
    image_data = image_data.split(',')[1]  # Remove data:image/png;base64, prefix
    image_bytes = base64.b64decode(image_data)
    
    # Open image with PIL
    img = Image.open(io.BytesIO(image_bytes))
    
    # Crop the image
    cropped = img.crop((x, y, x + width, y + height))
    
    # Resize to target dimensions
    resized = cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    # Save to bytes buffer
    output = io.BytesIO()
    resized.save(output, format='PNG')
    output.seek(0)
    
    return send_file(output, mimetype='image/png', as_attachment=True, 
                     download_name=f'cropped_{target_width}x{target_height}.png')

if __name__ == '__main__':
    app.run(debug=True)