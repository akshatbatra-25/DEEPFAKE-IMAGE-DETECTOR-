# inference.py (auto-detect fake index and display both probs)
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any
from PIL import Image
import torch
from torchvision import transforms
from facenet_pytorch import MTCNN

BASE = Path(__file__).resolve().parent
MODEL_TS_PATH = BASE / "models" / "final_model_best_epoch0_ts.pt"
IDX_MAP_PATH = BASE / "models" / "idx_to_class.json"
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if not MODEL_TS_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_TS_PATH}")
print(f"[inference] Loading TorchScript model from {MODEL_TS_PATH} on {DEVICE}")
_model_ts = torch.jit.load(str(MODEL_TS_PATH), map_location=DEVICE)
_model_ts.eval()

# Load optional mapping
idx_to_class = None
fake_index = None
if IDX_MAP_PATH.exists():
    try:
        idx_to_class = json.loads(IDX_MAP_PATH.read_text())
        idx_to_class = {int(k): v for k, v in idx_to_class.items()}
        print("[inference] idx_to_class:", idx_to_class)
        # find which index mentions 'fake'
        for k, v in idx_to_class.items():
            if isinstance(v, str) and 'fake' in v.lower():
                fake_index = int(k)
                break
        if fake_index is None:
            print("[inference] Could not auto-find 'fake' in idx_to_class.json; leaving auto-detect off.")
        else:
            print(f"[inference] auto-detected fake_index = {fake_index}")
    except Exception as e:
        print("[inference] failed to read idx_to_class.json:", e)

mtcnn = MTCNN(keep_all=True, device=DEVICE)
tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def _softmax_probs(pil_crop: Image.Image):
    inp = tf(pil_crop).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = _model_ts(inp)
        probs = torch.nn.functional.softmax(out, dim=1)[0].cpu().numpy().tolist()
    return probs

def predict_faces(pil_image: Image.Image) -> Tuple[List[Dict[str, Any]], Image.Image]:
    boxes, _ = mtcnn.detect(pil_image)
    annotated = pil_image.copy()
    results = []

    if boxes is None:
        probs = _softmax_probs(pil_image)
        # determine percent_fake using detected fake_index if available, else show both
        if fake_index is not None and fake_index < len(probs):
            percent_fake = float(probs[fake_index])
            # percent_real = sum of other indexes? if only two classes use other index
            other_idx = 1 - fake_index if len(probs) >= 2 else None
            percent_real = float(probs[other_idx]) if other_idx is not None else (1.0 - percent_fake)
        else:
            percent_fake = float(probs[0]) if len(probs) > 0 else 0.0
            percent_real = float(probs[1]) if len(probs) > 1 else (1.0 - percent_fake)
        results.append({"box": None, "probs": probs, "percent_fake": percent_fake, "percent_real": percent_real})
        return results, annotated

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = pil_image.crop((x1,y1,x2,y2))
        probs = _softmax_probs(crop)
        if fake_index is not None and fake_index < len(probs):
            percent_fake = float(probs[fake_index])
            other_idx = 1 - fake_index if len(probs) >= 2 else None
            percent_real = float(probs[other_idx]) if other_idx is not None else (1.0 - percent_fake)
        else:
            percent_fake = float(probs[0]) if len(probs) > 0 else 0.0
            percent_real = float(probs[1]) if len(probs) > 1 else (1.0 - percent_fake)
        results.append({"box":[x1,y1,x2,y2], "probs": probs, "percent_fake": percent_fake, "percent_real": percent_real})
    return results, annotated
