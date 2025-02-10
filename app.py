import os
import io
import base64
from flask import Flask, render_template, request, jsonify, url_for
import numpy as np
import cv2
from PIL import Image
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from werkzeug.utils import secure_filename

if not os.path.exists('static'):
    os.makedirs('static')

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- Helper function to load an image from a file path or data URL ---
def load_image(image_source):
    if image_source.startswith("data:"):
        base64_str = image_source.split(",", 1)[1]
        image_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    else:
        # Convert the URL (e.g. "/static/uploads/filename.png") into a file system path.
        file_path = os.path.join(os.getcwd(), image_source.lstrip("/"))
        return Image.open(file_path).convert("RGB")

# --- Resize (by scaling) and Censorship ---
def adjust_and_censor(image):
    """
    Resize the image to a square by scaling (stretching/compressing) to 256x256.
    Then, for each pixel (x,y) in the resized image, if y > (255 - x)
    (i.e. if the pixel is below the diagonal from bottom left to top right),
    set that pixel to white. No diagonal line is drawn.
    """
    size = 256
    image_square = image.resize((size, size), Image.LANCZOS)
    image_np = np.array(image_square)
    for y in range(size):
        for x in range(size):
            if y > (size - 1 - x):
                image_np[y, x] = [255, 255, 255]
    return Image.fromarray(image_np)

def extract_unique_colors(image_np, num_colors=10):
    pixels = image_np.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, random_state=0).fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_.reshape(image_np.shape[:2])
    return colors, labels

def interpolate_roc(FPR, TPR, smooth=True, window_length=11, polyorder=3):
    sorted_indices = np.argsort(FPR)
    FPR_sorted = FPR[sorted_indices]
    TPR_sorted = TPR[sorted_indices]
    FPR_unique, indices = np.unique(FPR_sorted, return_index=True)
    TPR_unique = TPR_sorted[indices]
    f_interp = interp1d(
        FPR_unique,
        TPR_unique,
        kind='linear',
        bounds_error=False,
        fill_value=(TPR_unique[0], TPR_unique[-1])
    )
    FPR_interp = np.linspace(0, 1, num=1000)
    TPR_interp = f_interp(FPR_interp)
    TPR_interp = np.clip(TPR_interp, 0, 1)
    FPR_interp = np.clip(FPR_interp, 0, 1)
    if smooth:
        if window_length % 2 == 0:
            window_length += 1
        TPR_interp = savgol_filter(TPR_interp, window_length, polyorder)
    return FPR_interp, TPR_interp

def extract_roc_data_from_colors(image_np, labels, selected_color_indices):
    height, width = image_np.shape[:2]
    mask = np.isin(labels, selected_color_indices).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask_filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    coords = np.column_stack(np.where(mask_filled))
    if coords.size == 0:
        raise ValueError("No pixels found for the selected colors.")
    TPR = 1 - (coords[:, 0] / height)
    FPR = coords[:, 1] / width
    TPR = np.clip(TPR, 0, 1)
    FPR = np.clip(FPR, 0, 1)
    FPR_interp, TPR_interp = interpolate_roc(FPR, TPR)
    return FPR_interp, TPR_interp

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-image', methods=['POST'])
def upload_image():
    image_file = request.files['image']
    filename = secure_filename(image_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_file.save(filepath)
    image_url = url_for('static', filename='uploads/' + filename)
    return jsonify({'imageUrl': image_url})

# In these endpoints, we always apply adjust_and_censor() (i.e. scaling + censorship)
@app.route('/extract-colors', methods=['POST'])
def extract_colors():
    data = request.json
    image_path = data['imagePath']
    num_colors = int(data.get('numColors', 5))
    image = load_image(image_path)
    image = adjust_and_censor(image)
    image_np = np.array(image)
    colors, _ = extract_unique_colors(image_np, num_colors)
    return jsonify({'colors': colors.tolist()})

@app.route('/display-color-options', methods=['POST'])
def display_color_options_endpoint():
    data = request.json
    image_path = data['imagePath']
    num_colors = int(data.get('numColors', 5))
    image = load_image(image_path)
    image = adjust_and_censor(image)
    image_np = np.array(image)
    colors, labels = extract_unique_colors(image_np, num_colors)
    option_images = []
    for i in range(len(colors)):
        mask = (labels == i)
        mask_image = np.zeros_like(image_np)
        mask_image[mask] = image_np[mask]
        buffered = io.BytesIO()
        img = Image.fromarray(mask_image)
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        option_images.append({"code": i, "img": "data:image/png;base64," + img_str})
    return jsonify({'options': option_images})

@app.route('/extract-roc', methods=['POST'])
def extract_roc():
    data = request.json
    image_path = data['imagePath']
    selected_color_indices = data['selectedColorIndices']
    num_colors = int(data.get('numColors', 5))
    image = load_image(image_path)
    image = adjust_and_censor(image)
    image_np = np.array(image)
    _, labels = extract_unique_colors(image_np, num_colors)
    FPR, TPR = extract_roc_data_from_colors(image_np, labels, selected_color_indices)
    # Remove spikes by enforcing TPR monotonicity.
    TPR = np.maximum.accumulate(TPR)
    return jsonify({'FPR': FPR.tolist(), 'TPR': TPR.tolist()})

@app.route('/compute-optimal', methods=['POST'])
def compute_optimal():
    data = request.json
    FPR = np.array(data['FPR'])
    TPR = np.array(data['TPR'])
    mode = data.get('mode', 'optimizer')
    youden_index = TPR + (1 - FPR) - 1
    if mode == 'optimizer':
        method = data.get('method', 'youden')
        if method == 'youden':
            metric_array = TPR + (1 - FPR) - 1
            optimal_idx = np.argmax(metric_array)
        elif method == 'distance':
            metric_array = np.sqrt(FPR**2 + (1 - TPR)**2)
            optimal_idx = np.argmin(metric_array)
        elif method == 'sensitivity+specificity':
            metric_array = TPR + (1 - FPR)
            optimal_idx = np.argmax(metric_array)
        elif method == 'sensitivity*specificity':
            metric_array = TPR * (1 - FPR)
            optimal_idx = np.argmax(metric_array)
        elif method in ['accuracy', 'f1_score']:
            try:
                total_positives = int(data['totalPositives'])
                total_negatives = int(data['totalNegatives'])
            except Exception:
                return jsonify({"error": "Total positives and negatives are required for accuracy and f1_score."}), 400
            metric_array = np.zeros_like(TPR)
            for i in range(len(TPR)):
                TP = TPR[i] * total_positives
                FP = FPR[i] * total_negatives
                TN = (1 - FPR[i]) * total_negatives
                FN = (1 - TPR[i]) * total_positives
                if method == 'accuracy':
                    metric_array[i] = (TP + TN) / (total_positives + total_negatives)
                else:
                    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                    recall = TPR[i]
                    metric_array[i] = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
            optimal_idx = np.argmax(metric_array)
        else:
            return jsonify({"error": f"Unknown optimizer method: {method}"}), 400

        tie_indices = np.where(np.abs(metric_array - metric_array[optimal_idx]) < 1e-6)[0]
        if len(tie_indices) > 1:
            candidate_youden = youden_index[tie_indices]
            optimal_idx = tie_indices[np.argmin(candidate_youden)]
        optimal_sensitivity = TPR[optimal_idx]
        optimal_specificity = 1 - FPR[optimal_idx]
        if method in ['accuracy', 'f1_score']:
            computed_metric = metric_array[optimal_idx]
        else:
            if method == 'youden':
                computed_metric = TPR[optimal_idx] + (1 - FPR[optimal_idx]) - 1
            elif method == 'distance':
                computed_metric = np.sqrt(FPR[optimal_idx]**2 + (1 - TPR[optimal_idx])**2)
            elif method == 'sensitivity+specificity':
                computed_metric = TPR[optimal_idx] + (1 - FPR[optimal_idx])
            elif method == 'sensitivity*specificity':
                computed_metric = TPR[optimal_idx] * (1 - FPR[optimal_idx])
        response = {
            "optimalSensitivity": float(optimal_sensitivity),
            "optimalSpecificity": float(optimal_specificity),
            "computedMetric": float(computed_metric)
        }
        return jsonify(response)
    elif mode == 'target':
        targetMetric = data.get('targetMetric')
        try:
            targetValue = float(data.get('targetValue'))
        except Exception:
            return jsonify({"error": "A valid target value is required."}), 400
        if targetMetric in ['accuracy', 'f1_score']:
            try:
                total_positives = int(data['totalPositives'])
                total_negatives = int(data['totalNegatives'])
            except Exception:
                return jsonify({"error": "Total positives and negatives are required for accuracy and f1_score."}), 400
        metric_array = np.zeros_like(TPR)
        if targetMetric == 'sensitivity':
            metric_array = TPR
        elif targetMetric == 'specificity':
            metric_array = 1 - FPR
        elif targetMetric == 'accuracy':
            for i in range(len(TPR)):
                TP = TPR[i] * total_positives
                FP = FPR[i] * total_negatives
                TN = (1 - FPR[i]) * total_negatives
                FN = (1 - TPR[i]) * total_positives
                metric_array[i] = (TP + TN) / (total_positives + total_negatives)
        elif targetMetric == 'f1_score':
            for i in range(len(TPR)):
                TP = TPR[i] * total_positives
                FP = FPR[i] * total_negatives
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TPR[i]
                metric_array[i] = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        else:
            return jsonify({"error": "Unknown target metric."}), 400
        diff_array = np.abs(metric_array - targetValue)
        optimal_idx = np.argmin(diff_array)
        tie_indices = np.where(np.abs(diff_array - diff_array[optimal_idx]) < 1e-6)[0]
        if len(tie_indices) > 1:
            candidate_youden = youden_index[tie_indices]
            optimal_idx = tie_indices[np.argmin(candidate_youden)]
        optimal_sensitivity = TPR[optimal_idx]
        optimal_specificity = 1 - FPR[optimal_idx]
        computed_metric = metric_array[optimal_idx]
        response = {
            "optimalSensitivity": float(optimal_sensitivity),
            "optimalSpecificity": float(optimal_specificity),
            "computedMetric": float(computed_metric)
        }
        return jsonify(response)
    else:
        return jsonify({"error": "Unknown mode."}), 400

if __name__ == '__main__':
    import webbrowser
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        webbrowser.open("http://127.0.0.1:5000/")
    app.run(debug=False)
