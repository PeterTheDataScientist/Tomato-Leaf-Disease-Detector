import json
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import HybridCNNTransformer

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "hybrid_tomato_leaf_best_lowlight_hardened.pth"
CLASS_NAMES_PATH = "class_names.json"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMG_SIZE = 224

# -----------------------------
# Helpers
# -----------------------------
def pretty_label(raw: str) -> str:
    # "Tomato___Tomato_Yellow_Leaf_Curl_Virus" -> "Tomato Yellow Leaf Curl Virus"
    s = raw.replace("Tomato___", "")
    s = s.replace("_", " ")
    s = " ".join(s.split())
    return s

def confidence_level(conf: float) -> str:
    if conf >= 0.85:
        return "high"
    elif conf >= 0.60:
        return "medium"
    return "low"

# -----------------------------
# Preprocess (must match eval)
# -----------------------------
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# -----------------------------
# Load class names
# -----------------------------
@st.cache_data
def load_class_names(path: str):
    with open(path, "r") as f:
        idx_to_class = json.load(f)
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}  # ensure int keys
    return [idx_to_class[i] for i in range(len(idx_to_class))]

# -----------------------------
# Load model (cached)
# -----------------------------
@st.cache_resource
def load_model(model_path: str, num_classes: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridCNNTransformer(num_classes=num_classes, img_size=IMG_SIZE)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device

def predict(model, device, pil_image: Image.Image, class_names, topk=5):
    img = pil_image.convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)  # [1,3,224,224]

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()

    topk = min(topk, len(class_names))
    confs, idxs = torch.topk(probs, k=topk)

    results = []
    for c, i in zip(confs.tolist(), idxs.tolist()):
        results.append((class_names[i], float(c)))
    return results

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Tomato Leaf Disease Detector",
    page_icon="🍅",
    layout="centered"
)

st.title("🍅 Tomato Leaf Disease Detector")
st.write(
    "Upload a tomato leaf image. The app predicts the disease class using a hybrid CNN–Transformer model."
)

# Load assets
class_names = load_class_names(CLASS_NAMES_PATH)
model, device = load_model(MODEL_PATH, num_classes=len(class_names))

st.caption(f"Running on: **{device}** | Classes: **{len(class_names)}**")

uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns([1, 1])

if uploaded is not None:
    image = Image.open(uploaded)

    with col1:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction")
        results = predict(model, device, image, class_names, topk=5)

        best_label, best_conf = results[0]
        best_label_pretty = pretty_label(best_label)

        is_healthy = ("healthy" in best_label.lower())

        # Healthy vs Diseased badge
        if is_healthy:
            st.success(f"✅ Result: **{best_label_pretty}**")
        else:
            st.error(f"⚠️ Result: **{best_label_pretty}**")

        # Confidence messaging
        lvl = confidence_level(best_conf)
        if lvl == "high":
            st.info("Confidence: **High** ✅")
        elif lvl == "medium":
            st.warning("Confidence: **Medium** — consider retaking the photo in better lighting.")
        else:
            st.error("Confidence: **Low** — prediction may be unreliable. Try a clearer photo (better light, closer, in focus).")

        st.progress(min(max(best_conf, 0.0), 1.0))
        st.write(f"Model confidence: **{best_conf:.2%}**")

        st.write("Top-5 predictions:")
        for label, conf in results:
            st.write(f"- {pretty_label(label)}: {conf:.2%}")

# -----------------------------
# Tips section (in app.py)
# -----------------------------
st.markdown("---")
st.subheader("📸 Tips for best results")
st.write("- Make the leaf fill most of the frame (move closer instead of zooming).")
st.write("- Use natural daylight when possible; avoid heavy shadows over the leaf.")
st.write("- Keep the leaf in focus (avoid motion blur).")
st.write("- If lighting is very dim, retake the photo with more light.")
st.write("- Try taking 2–3 photos from slightly different angles.")

# -----------------------------
# Simple model card (in app.py)
# -----------------------------
with st.expander("ℹ️ Model Card (Project Info)"):
    st.write("**Model type:** Hybrid CNN–Transformer (ResNet18 backbone + Transformer encoder + feature fusion)")
    st.write("**Classes:** 10 tomato leaf conditions (including healthy)")
    st.write("**Training data:** Kaggle Tomato Leaf Disease dataset (train split), Kaggle test split used for evaluation")
    st.write("**Robustness:** Trained with realistic augmentations (low-light, blur, noise, perspective, resolution simulation) + MixUp + label smoothing")
    st.write("**Best use case:** Tomato leaf close-up images under typical farming conditions")
    st.write("**Limitations:** Extreme occlusion, non-leaf images, or very poor lighting may reduce reliability.")
