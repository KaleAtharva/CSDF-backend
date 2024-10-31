from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageChops, ImageDraw, ImageEnhance
import os
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def ela_analysis(image_path, output_path, quality=90):
    """
    Perform Error Level Analysis on the image
    """
    try:
        # Open original image
        original = Image.open(image_path).convert('RGB')
        
        # Save temporary compressed version
        temp_path = "temp.jpg"
        original.save(temp_path, 'JPEG', quality=quality)
        compressed = Image.open(temp_path).convert('RGB')
        
        # Calculate difference and amplify
        diff = ImageChops.difference(original, compressed)
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        
        scale = 255.0 / max_diff
        diff = ImageEnhance.Brightness(diff).enhance(scale)
        
        # Convert to grayscale for better analysis
        diff_gray = diff.convert('L')
        
        # Save ELA result
        diff.save(output_path, 'JPEG')
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return diff_gray

    except Exception as e:
        print(f"Error during ELA analysis: {e}")
        return None

def detect_forgery(input_path, output_path, block_size=16, threshold_multiplier=5):
    """
    Detect potential forgery in the image using ELA
    """
    try:
        # Perform ELA analysis
        ela_image = ela_analysis(input_path, output_path)
        if ela_image is None:
            return False, None

        # Convert to numpy array for processing
        ela_array = np.array(ela_image)
        height, width = ela_array.shape
        
        # Calculate local statistics
        tampered_blocks = []
        original = Image.open(input_path).convert('RGB')
        draw = ImageDraw.Draw(original)
        
        # Analyze image in blocks
        for y in range(0, height - block_size, block_size):
            for x in range(0, width - block_size, block_size):
                block = ela_array[y:y+block_size, x:x+block_size]
                
                # Calculate block statistics
                block_mean = np.mean(block)
                block_std = np.std(block)
                
                # Dynamic thresholding
                local_threshold = block_mean + (block_std * threshold_multiplier)
                
                # Check for anomalies
                if np.max(block) > local_threshold and block_std > 10:
                    tampered_blocks.append((x, y))
                    draw.rectangle(
                        [(x, y), (x + block_size, y + block_size)],
                        outline="red",
                        width=2
                    )

        # Save highlighted image if tampering detected
        highlighted_path = f"highlighted_{os.path.basename(output_path)}"
        full_highlighted_path = os.path.join(OUTPUT_FOLDER, highlighted_path)
        original.save(full_highlighted_path)
        
        # Return results
        is_tampered = len(tampered_blocks) >= 3  # Require at least 3 suspicious blocks
        return is_tampered, highlighted_path

    except Exception as e:
        print(f"Error in forgery detection: {e}")
        return False, None

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Save uploaded file
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    output_path = os.path.join(OUTPUT_FOLDER, f"ela_{file.filename}")
    file.save(input_path)

    # Perform forgery detection
    is_tampered, highlighted_path = detect_forgery(input_path, output_path)

    # Clean up uploaded file
    if os.path.exists(input_path):
        os.remove(input_path)

    if is_tampered and highlighted_path:
        return send_file(
            os.path.join(OUTPUT_FOLDER, highlighted_path),
            mimetype='image/jpeg'
        )
    else:
        return jsonify({"message": "No forgery detected"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)