# ROC Extractor

The **ROC Curve Analyzer** is a Flask-based web application with python back-end for extracting and digitalizing ROC curve plots from images. The point estimates and/or optimal point of the curve can be computed for use cases such as data extraction in DTA meta-analyses. 
The app offers the following features:

- **Image Upload:** Upload PNG or JPEG images.
- **Optional Cropping:** Use the built-in cropping tool to select the plotting region (0 to 1).
- **Automatic Resizing & Pre-processing:**  
  - The image is automatically scaled to a square.  
  - Pixels below the diagonal (running from the bottom left to the top right) are censored to decrease noise.
- **Color Clustering:** Extract unique colors using KMeans clustering. In case of difficulty in finding desired colors, it allows visualization of raw maps of the clustering output for better confirmation.
- **ROC Curve Extraction**  
  - Extract (FPR, TPR) data points from the image. noise reduction steps are also performed in this step to address clustering errors and remove spikes. 
- **Computing optimal points:**
  - Compute an optimal cutoff point using several methods (e.g., Youden’s index, distance, sensitivity+specificity, accuracy, F1 score).
  - It also allows for targeting a point with a specific metric (sensitivity, specificity, or accuracy)

## Installation
1. **Clone the repository:**

   ```bash
   git clone https://github.com/Payam-PJD/ROC-extractor.git
   cd ROC-extractor

2. **Install dependencies:**

  -Using pip:
   ```bash
   pip install -r requirements.txt
```
  -Using conda:
   ```bash
   conda install -c conda-forge flask opencv numpy scipy pillow scikit-learn
```
## Running the app
After installing the dependencies, start the Flask development server by running:
```bash
python3 app.py
```
This will launch the server and automatically open your default browser at http://127.0.0.1:5000/.
## Manual
1. Upload:
 Select and upload an image (PNG or JPEG).

2. Cropping (Optional):
  Check the "Enable Cropping (optional)" box if you wish to adjust the image.
  Use the cropping tool to select your desired region, then click Crop.
  If you do not use cropping, the app continues with the full uploaded image.

3. Color clustering:
  Enter the number of unique colors to extract and click Extract Colors. The extracted color options will be displayed.
  The difficult number of unique colors is set to 5. If your ROC curve has fewer colors, please decrease it. If it has several curves with different colors, increase the number of colors.

4. Curve selection:
  Click on the color box (or use raw options) to select the ROC curve color.

5. ROC preview:
  Click Preview ROC to display the extracted ROC curve.

6. Optimization:
  Select an optimization mode and provide any additional parameters if required, then click Compute Optimal Point. The optimal point is computed and shown on the ROC curve.
