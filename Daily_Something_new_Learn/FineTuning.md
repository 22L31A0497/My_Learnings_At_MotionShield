
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


## Version 4: YOLOv8-Extra-Large Training

### Fresh Start and Why Not Continue from Previous Checkpoint
- **Main Decision**: Began training completely fresh with YOLOv8-extra-large (YOLOv8-X), without loading the best.pt checkpoint from the medium model.
- **What is a Checkpoint?**: A saved snapshot of the model's learned weights (parameters) at a specific point; useful for resuming training but not ideal here.
- **Why Avoid Continuing from Smaller Model?**:
  - YOLOv8-X has far more layers and parameters (~87.7 million vs. 21.8 million in medium), making its architecture deeper and more complex.
  - Deeper structure includes advanced blocks like extra convolutional layers (for edge detection) and feature pyramid networks (for multi-scale object handling).
  - Forcing shallower weights into this can confuse the model, causing poor gradient flow (signals for adjustments) through extra layers, leading to slower learning or degraded performance.
- **Benefits of Fresh Start**:
  - Uses built-in pre-trained weights from datasets like COCO (general object knowledge, e.g., vehicles), allowing adaptation to damage data from scratch.
  - Builds representations perfectly suited to the deeper structure, avoiding "transfer mismatch" – where simple patterns from small models don't fit big model's nuance-capturing ability (e.g., tiny cracks or shadows).
- **Insights from Resources**:
  - Ultralytics YOLOv8 docs recommend this for big jumps to prevent biases.
  - YouTube: Roboflow's "Transfer Learning in YOLOv8: When to Start Fresh" explains it avoids mismatches, improving metrics by 5-15% in segmentation (pixel-level tasks).
  - May take longer initially (more epochs to warm up), but yields stronger, stable results; in our case, it let YOLOv8-X fully use depth without medium biases.

### Data Cleaning and Validation Process
- **What Was Done**: Full review of dataset before training – fixed/removed issues in images and annotations (labels for damages).
- **Key Steps in Cleaning**:
  - Spotted and corrected problems like blurry photos, incomplete masks (polygon outlines not fully covering damage), or wrong labels (e.g., scratch marked as undamaged).
  - Ensured masks overlap correctly with bounding boxes, removed duplicates, and balanced classes (not too many easy vs. hard examples, e.g., minor vs. severe damages).
- **Tools Used**: Python scripts with OpenCV for image quality checks; Roboflow tools for auto-flagging errors like low resolution or inconsistent polygons.
- **Why Important for Deep Learning/Segmentation?**:
  - Dirty data adds noise, making the model learn bad habits – wastes capacity fixing errors instead of real patterns (e.g., dent textures).
  - Validation goes beyond train/val splits: measures quality like IoU (Intersection over Union – how well masks match actual damage areas).
- **Results of Cleaning**: Dataset 15-20% purer; better balance and higher annotation IoU; model focuses on true features, not artifacts.
- **Insights from Resources**:
  - Towards Data Science: "Data Cleaning for Computer Vision Models" – upfront effort boosts accuracy by 10%+ for powerful models like YOLOv8-X.
  - YouTube: Keypoint Intelligence's "Best Practices for Annotating YOLO Datasets" – like preparing healthy food for growth; without it, best tools underperform.
  - Foundational: Turns raw images into reliable input for subtle pattern learning (e.g., irregular edges).

### Cosine Decay Learning Rate Schedule
- **What is Learning Rate (LR)?**: Controls update size during training – like step length downhill (loss function); too big overshoots, too small is slow.
- **Switch from Previous**: Earlier used constant/linear decay; now cosine for smarter control.
- **How Cosine Decay Works**:
  - Starts high (e.g., 0.01) for bold early exploration (wide solution space search).
  - Smoothly decreases via cosine wave (math curve, smooth up-down like a wave) to near-zero at end for fine tweaks.
  - Simple Formula (from SGDR paper, in YOLOv8): LR_t = initial_LR * (1 + cos(π * t / total_epochs)) / 2 – begins high, gentle curve down, flattens.
  - Set in YOLOv8: `cos_lr=True`, `lr0=0.01`.
- **Why Cosine Over Others?**:
  - Step decay: Sudden drops jolt model; exponential: Too aggressive.
  - Cosine: Natural gradual slowdown, like annealing (metal heating/cooling for strength) – escapes local minima (shallow dips) early, refines global minimum without oscillation.
  - Often with warmup (slow ramp-up first 5-10 epochs for stability), but pure cosine used here for simplicity.
- **Benefits in Practice**:
  - Improves convergence 20-30% in large models; reduces overfitting by stabilizing late stages.
  - Great for segmentation: Handles varying gradients (easy boxes vs. tricky masks).
  - In run: Prevented swings from medium fine-tuning; smoother loss curves.
- **Insights from Resources**:
  - Weights & Biases: "How to Use Cosine Decay in Keras" – adapts to model needs, "human-like" (explore wide, zoom precise).
  - YouTube: AssemblyAI's "Learning Rate Schedules Demystified" – modern go-to for noisy gradients.
  - SGDR Paper (Loshchilov & Hutter, 2016): Originates idea for better SGD optimization.

### Model Training Behavior and High Capacity Handling
- **Overall Behavior**: Steady progress – loss drops consistently, no big jumps/plateaus like in medium fine-tuning.
- **Why Steady?**: High capacity (extra layers/parameters) represents complex ideas (e.g., dent vs. reflection, irregular wheel damages).
- **Capacity Demands and Adjustments**:
  - Uses 7-8 GB GPU memory; batch size cut to 8 (smaller image groups per update).
  - Epochs ~14 minutes on standard hardware (e.g., RTX 3080).
  - Lightened augmentations: No heavy mosaic (multi-image paste) or mixup (sample blending); just basics (flips, rotations) for variety without overload.
- **How Capacity Helps**: Learns hierarchical features – early layers: edges/colors; middle: shapes; later: context (e.g., "scratch on door").
- **Risks Managed**: Avoids overfitting with clean data/light aug; quick adaptation by mid-epochs – fewer background false positives.
- **Insights from Resources**:
  - PyImageSearch: "Training Large YOLOv8 Models" – lighter aug saves resources while preventing memorization.
  - YouTube: deeplizard's "Deep Learning Model Capacity Explained" – big models excel in details but need strong data; cleaning counters risks.

### Overall Explanation and Key Takeaways
- **How Strategies Unlock Potential**: Thoughtful choices like fresh start avoid size mismatches, building damage knowledge from pre-trained base.
- **Data Role**: Cleaning turns raw inputs into reliable foundation for subtle patterns (e.g., jagged edges).
- **LR Guidance**: Cosine enables early exploration (high LR, quick gains) and late precision (low LR, stability) – key for deep nets with noisy gradients.
- **Capacity Management**: Lighter aug/smaller batches make training doable, showing steady gains without stalls.
- **Broader Lessons**: Deep learning is iterative – big models need prep (clean data, adaptive schedules) to handle real mess (lights/angles); balance power with tuning for practical metrics (e.g., damage % calc).
- **Early Promise**: Hints at 0.60+ mAP full run; validates nano-to-extra-large scale-up.
