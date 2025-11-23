# app.py
import os
import json
from flask import Flask, render_template, request, send_file, abort
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn

# ---------------------------
# Config
# ---------------------------
MODEL_PATH = "best_checkpoint_full_fast.pth"   # <-- your checkpoint filename
IDX_PATH = "idx_to_class.json"                 # <-- mapping saved during training
UPLOADED_NAME = "uploaded.jpg"                 # served at /uploaded.jpg
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224

# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__)

# ---------------------------
# Load idx_to_class mapping
# ---------------------------
if not os.path.exists(IDX_PATH):
    raise FileNotFoundError(f"Missing {IDX_PATH} in app folder. Copy idx_to_class.json here.")

with open(IDX_PATH, "r") as fh:
    raw_map = json.load(fh)

# Normalize mapping keys to stringified ints (consistent usage)
idx_to_class = {str(k): v for k, v in raw_map.items()}
num_classes = len(idx_to_class)
if num_classes == 0:
    raise ValueError("idx_to_class.json seems empty or invalid.")

# ---------------------------
# Build model architecture (must match training)
# ---------------------------
# We used ResNet50 with a custom head: Linear(in_f,512)->ReLU->Dropout(0.4)->Linear(512,num_classes)
model = models.resnet50(weights=None)   # do not load pretrained here
in_f = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(in_f, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.4),
    nn.Linear(512, num_classes)
)

# ---------------------------
# Load checkpoint correctly
# ---------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Missing {MODEL_PATH} in app folder. Copy checkpoint here.")

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

# checkpoint may be a dict with 'model_state' or may already be a state_dict
if isinstance(ckpt, dict) and 'model_state' in ckpt:
    state_dict = ckpt['model_state']
elif isinstance(ckpt, dict) and 'model' in ckpt:
    state_dict = ckpt['model']
else:
    state_dict = ckpt

# Ensure keys are correct type (some saving workflows store stringified keys)
try:
    model.load_state_dict(state_dict)
except RuntimeError as e:
    # try to fix possible "module." prefix mismatch or key type mismatches
    new_state = {}
    for k, v in state_dict.items():
        new_key = k
        if k.startswith("module."):
            new_key = k[len("module."):]
        new_state[new_key] = v
    model.load_state_dict(new_state)

model.to(DEVICE)
model.eval()

# ---------------------------
# Transforms (match training normalization)
# ---------------------------
transform = transforms.Compose([
    transforms.Resize(int(IMAGE_SIZE * 1.14)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ---------------------------
# Helper predict function
# ---------------------------
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    # Build ordered mapping label->probability (use string keys from idx_to_class)
    # idx_to_class maps "0"/"1" -> label names
    out = {}
    for i in range(len(probs)):
        key = str(i)
        label = idx_to_class.get(key, key)
        out[label] = float(probs[i])
    # Sort by probability descending for nicer display (optional)
    out = dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))
    return out

# ---------------------------
# Route to serve the latest uploaded image at /uploaded.jpg
# ---------------------------
@app.route("/uploaded.jpg")
def serve_uploaded():
    if not os.path.exists(UPLOADED_NAME):
        abort(404)
    return send_file(UPLOADED_NAME, mimetype='image/jpeg')

# ---------------------------
# Home page
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------------------
# Predict page (handles GET and POST)
# ---------------------------
@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    prediction = None
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("predict.html", prediction=None)
        file = request.files["file"]
        if file.filename == "":
            return render_template("predict.html", prediction=None)

        # Save uploaded image to working dir with fixed name (overwritten each request)
        file.save(UPLOADED_NAME)

        # Run prediction
        prediction = predict(UPLOADED_NAME)

    return render_template("predict.html", prediction=prediction)

# ---------------------------
# Run the app
# ---------------------------
if __name__ == "__main__":
    # For local testing only. Use a proper WSGI server in production.
    app.run(debug=True, host="0.0.0.0", port=5000)