# Tomato Leaf Disease Detection: Hybrid CNN-Transformer

A robustness-engineered deep learning model that classifies ten tomato leaf conditions from a single phone photo, built to survive real field conditions (low light, budget cameras, blur, clutter) rather than only scoring well on clean laboratory images.

**Streamlit demo:** https://tomato-leaf-disease-detector-peter.streamlit.app/

This is my completed BSc dissertation (University of Zimbabwe, 2026) and the proof of concept for my ongoing, independent multi-crop research, ZimCropGuard.

## Architecture

A hybrid CNN-Transformer (a Transformer-augmented CNN). It is **not** a Vision Transformer. The description below matches `model.py` exactly.

- An ImageNet-pretrained **ResNet18** backbone is the feature extractor, producing a 512-channel 7x7 feature map.
- Two parallel branches read that map:
  - **CNN-global branch:** global average pooling followed by a linear projection to a 256-dimensional vector.
  - **Attention branch:** the 49 spatial cells are flattened into 49 tokens, each projected to 256 dimensions, with a learnable CLS token and positional embeddings prepended.
- A **4-layer Transformer encoder** (8 attention heads, GELU MLP, pre-norm residual blocks) reasons over the tokens. The embedding dimension (`embed_dim`) is 256.
- **Late fusion:** the CNN global vector is concatenated with the Transformer CLS output to form a 512-dimensional fused representation, which an MLP classifier head maps to the class logits.

Framework: **PyTorch only.** No TensorFlow, no Keras.

## Robustness

The shipped weights (`hybrid_tomato_leaf_best_lowlight_hardened.pth`) are the low-light-hardened model. It was trained with field-simulating augmentation (brightness and contrast variation, Gaussian noise and blur, perspective transforms, resolution simulation), MixUp, and label smoothing, so it degrades gracefully on real phone photos instead of collapsing off clean data.

## What's in this repo

This is an inference-plus-weights repository. It contains the app, the architecture, and the trained weights, not the training pipeline.

- `app.py`: Streamlit app for inference.
- `model.py`: the hybrid CNN-Transformer architecture.
- `hybrid_tomato_leaf_best_lowlight_hardened.pth`: trained hardened weights.
- `class_names.json`: class label mapping.
- `requirements.txt`: Python dependencies.

Note: `__pycache__/` should be gitignored and is not part of the repo. Add a `.gitignore` line for it:

```
__pycache__/
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Parameter count

The total parameter count is not asserted here. Print it directly from the model:

```python
from model import HybridCNNTransformer
model = HybridCNNTransformer(num_classes=10)
print(sum(p.numel() for p in model.parameters()))
```

## Research context

This tomato model is the completed BSc dissertation (2026) and the proof of concept beneath **ZimCropGuard**, ongoing independent research that scales the same robustness-first approach to five Zimbabwean staple crops and 21+ disease classes with a larger dual-backbone (EfficientNetB0 and ViT-B/16) architecture.

## Licence

No licence is included yet. An **MIT Licence** is recommended for this project: it is permissive, widely understood, and appropriate for a public research demo. Add a `LICENSE` file with the standard MIT text to apply it.
