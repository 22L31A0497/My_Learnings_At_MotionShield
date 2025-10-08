
# Vehicle Damage Detection with YOLOv8: Progress Documentation

This document tracks daily learnings and observations. All explanations use simple terms for clear understanding.

## Model Basics and Why We Use Them

YOLOv8 models come in sizes: nano (small, fast but limited), small, medium, large, and extra-large (big, powerful but slower). Larger models have more layers and parameters (like neurons in a brain), allowing them to learn complex patterns like tiny damages or varying light conditions. 

From internet resources (e.g., Ultralytics docs and YouTube: "YOLOv8 Explained" by Roboflow), smaller models are good starters but miss details; bigger ones capture finer features but need more data and compute power to avoid overfitting (memorizing instead of learning generally).

## Version 1: YOLOv8-Nano Baseline (Early September 2025)

### Setup and Training
- Model: Nano (~3.7 million parameters) – smallest size for quick testing.
- Goal: Basic segmentation of vehicles and damage areas.

### Observations
- Metrics: mAP@0.5 around 0.45, box precision 0.52, mask precision 0.48.
- Strengths: Trained fast (30 mins per epoch), handled simple cases well.
- Weaknesses: Missed small or hidden damages; high false negatives in low light.

### Key Learnings
Nano is like a basic tool – efficient but lacks depth. It couldn't extract enough features from images because of few layers. This matches YouTube videos (e.g., "Why Model Size Matters in YOLO" by AI Coffee Break), where small models work for prototypes but need upgrades for accuracy. Decision: Scale up for better learning capacity.

## Version 3: YOLOv8-Medium Upgrade (Mid-September 2025)

### Setup and Training
- Model: Medium (~21.8 million parameters) – 5x more than nano, deeper layers.
- Changes: Increased epochs to 100, added augmentations (image flips, brightness shifts).

### Observations
- Metrics: Improved mAP@0.5 to 0.62, box precision 0.68, mask precision 0.65 – big jump from V1.
- Strengths: Better at segmenting irregular damage shapes; reduced misses by 25%.
- Weaknesses: Still struggled with very fine scratches or overlapping parts.

### Key Learnings
Switching to medium added layers for better feature detection, like recognizing edges and textures. From online resources (e.g., Towards Data Science article "Scaling YOLO Models"), more parameters help in handling variations in vehicle types/angles. This version showed clear gains, proving bigger models learn more from the same data. No major overfitting seen with proper validation split.

## Recent Fine-Tuning on Medium Model (October 7-8, 2025 Update)

### Setup and Training
- Model: Same medium as V3, fine-tuned for two days with advanced params (lower learning rate 0.001, Adam optimizer, more epochs).
- Goal: Squeeze extra performance without changing model size.

### Observations
- Metrics: Mask precision up slightly (0.67), box precision up (0.70), fitness up (0.68) – small wins.
- Other metrics (recall, mAP) dipped below V3 levels (e.g., recall 0.60 vs. 0.65).
- Overall: No big improvements; model seemed "stuck" after initial gains.

### Key Learnings
Even with tweaks like better optimizers (Adam adapts learning rates dynamically, explained in YouTube: "Adam Optimizer in Deep Learning" by 3Blue1Brown-inspired channels), the medium model hit a limit. It lacks enough layers/parameters to learn subtle patterns beyond basics. This is common in deep learning – models plateau when capacity is insufficient (from Stack Overflow discussions and "Neural Network Depth" videos). Insight: Fine-tuning helps polish but can't fix under-capacity; time to scale up.

## Planned Next: Version 4 – YOLOv8-Extra-Large (Starting October 9, 2025)

### Setup Preview
- Model: Extra-large (~87.7 million parameters) – 4x more than medium, deepest architecture.
- Expectations: Train on GPU with batch size 8, monitor for overfitting with early stopping.

### Rationale and Learnings
After researching (e.g., Ultralytics benchmarks and YouTube: "YOLOv8 Large vs Small" by Murtaza's Workshop), extra-large models excel in segmentation tasks like damage detection due to richer feature maps. They handle complexity better, like varying damage sizes or backgrounds. Risks: Longer training (hours per epoch), higher VRAM needs (16GB+ recommended). Decision: This should boost metrics by 10-20% based on similar projects; will document overfitting checks and damage percentage calculations next.

## Overall Insights Across Versions
- Progression shows model size drives gains: Nano for speed, medium for balance, extra-large for precision.
- Metrics like mask precision measure segmentation accuracy (pixel-level), while box precision is for bounding boxes – both improved with depth.
- Common theme: Data quality + model capacity = better results. Future updates will add ablation studies (testing one change at a time).

## Resources Referenced
- YouTube: Roboflow (YOLOv8 segmentation tutorials), AI Coffee Break (model scaling).
- Websites: Ultralytics docs, Towards Data Science (optimizer effects).

# 🧠 Fine-Tuning a Neural Network

## 📌 What is Fine-Tuning?

Fine-tuning is the process of taking a pre-trained neural network and adapting it to a new, but related task. Instead of training from scratch, we reuse the learned features and adjust them slightly for our specific needs.

---

## 🎯 Why Fine-Tune?

- Saves time and computational resources.
- Works well when we have limited data.
- Leverages powerful features learned from large datasets.

---

## 🏗️ How Fine-Tuning Works

### 1. **Start with a Pre-trained Model**
   - Example: A CNN trained on ImageNet.
   - It has already learned general features like edges, textures, shapes.

### 2. **Freeze Early Layers**
   - These layers capture generic features.
   - Freezing means we don’t update their weights during training.

### 3. **Replace Final Layers**
   - Swap out the last few layers (usually fully connected ones).
   - Add new layers suited to your task (e.g., classification with different number of classes).

### 4. **Train the New Layers**
   - Only the new layers are trained initially.
   - Optionally, unfreeze some earlier layers later for deeper fine-tuning.

---

## 🧪 Example Workflow

```python
# Load pre-trained model
model = torchvision.models.resnet50(pretrained=True)

# Freeze early layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Train only the new layer
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
```

---

## 🧠 Transfer Learning vs Fine-Tuning

| Concept           | Transfer Learning                        | Fine-Tuning                              |
|------------------|------------------------------------------|------------------------------------------|
| Definition        | Using a pre-trained model as-is          | Slightly adjusting the pre-trained model |
| Layers Trained    | Usually only final layers                | Final layers + optionally earlier layers |
| Use Case          | When tasks are similar but not identical | When tasks are closely related           |

---

## ✅ Best Practices

- Use a model trained on a similar domain.
- Freeze layers wisely: early layers are more general, later ones are task-specific.
- Use a lower learning rate to avoid destroying learned features.

---

## 📚 Summary

Fine-tuning is a powerful technique in deep learning that allows us to adapt existing models to new tasks efficiently. It’s especially useful when data is scarce and training from scratch isn’t feasible.

---

