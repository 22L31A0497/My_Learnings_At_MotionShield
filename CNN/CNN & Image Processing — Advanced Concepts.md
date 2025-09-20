
# CNN & Image Processing — Advanced Concepts (Beginner Friendly)

## 🔑 Quick One-Liners

* **BN/LN:** Normalize activations to keep values steady → faster, stable training.
* **Residual/Skip:** Shortcuts that add input back → fix vanishing gradients in deep nets.
* **Attention (SE, CBAM, Self-Attention):** Focus on important features and locations.
* **Dilated Convs:** Wider view without more parameters → great for segmentation.
* **Deformable Convs:** Flexible sampling that follows object shapes.

---

## 1) Batch Normalization (BN) vs Layer Normalization (LN)

**What:**

* BN normalizes activations using the **statistics of the batch**.
* LN normalizes activations **per sample, across features**.

**Why:**

* Keeps training stable by preventing exploding/vanishing gradients.
* Allows using **larger learning rates** → faster convergence.

**Analogy:**
Like resetting everyone’s exam scores so the average is **0** and the spread is similar — this makes learning fair and smooth at each step.

**Mini-math (Batch Norm):**

$$
\mu_B = \frac{1}{m} \sum_i x_i, \quad 
\sigma_B^2 = \frac{1}{m} \sum_i (x_i - \mu_B)^2
$$

$$
\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad 
y = \gamma \hat{x} + \beta
$$

* During **training** → use batch mean/variance.
* During **inference** → use running averages.

**When to use:**

* **BN** → Standard in CNNs (ResNet, VGG).
* **LN** → Common in RNNs/Transformers (where batch size may vary).

**Code pattern (PyTorch):**

```python
nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU()
)
```

---

## 2) Residual and Skip Connections (ResNet, Dense Ideas)

**What:**

* Add a shortcut:

$$
\text{out} = F(x) + x
$$

* If input/output shapes differ → use a **1×1 conv** on the skip.

**Why:**

* Fixes **vanishing gradient** problem.
* Lets very deep networks (100+ layers) train properly.

**Analogy:**
Like a **flyover**: instead of stopping at every signal, cars (information) can directly skip traffic.

**Block Recipe (ResNet):**

* Conv → BN → ReLU → Conv → BN → **Add skip** → ReLU

**Dense Connections Idea:**

* DenseNet connects **each layer to all future layers** → encourages feature reuse and efficiency.

**Use Cases:**

* Vision models (ResNet)
* Skip ideas also inspire Transformers.

---

## 3) Attention in Vision (SE, CBAM, Self-Attention/ViT)

**What:**
Learn to **focus** on important features (channels) and locations (spatial).

**Why:**

* Improves accuracy by ignoring noise.
* Makes models more interpretable.

---

### 🔹 SE (Squeeze-and-Excitation)

* Global Average Pooling per channel → small MLP → sigmoid weights → rescale channels.
* **Analogy:** Like adjusting the **volume mixer**: boost important instruments, lower others.

---

### 🔹 CBAM (Convolutional Block Attention Module)

* **Channel attention**: choose important features.
* **Spatial attention**: highlight important locations.
* **Analogy:** First choose the right instruments, then shine a **spotlight** on the stage area.

---

### 🔹 Self-Attention (Vision Transformers)

* Each patch attends to all other patches → captures **global context**.
* Different from CNNs, which only look locally.

---

## 4) Dilated (Atrous) Convolutions

**What:**

* Insert gaps in the kernel to **expand receptive field** without adding parameters.
* Dilation rate:

  * r = 1 → normal conv
  * r = 2 → skips 1 pixel, covers wider area

**Why:**

* Useful in **dense prediction tasks** (segmentation, depth estimation).

**Analogy:**
Like **zooming out** on a camera → same lens, but capture a wider view.

**DeepLab models:**

* Use dilated convs to preserve resolution + context.
* Often combined with **ASPP (Atrous Spatial Pyramid Pooling)** for multi-scale vision.

---

## 5) Deformable Convolutions

**What:**

* Standard conv → samples at fixed grid points.
* Deformable conv → learns **offsets** to shift sampling locations.

**Formula Idea:**

* Normal: sample at $p_n$
* Deformable: sample at $p_n + \Delta p_n$

**Why:**

* Handles irregular shapes, poses, perspective changes.
* Adds small overhead, but improves accuracy in **detection/segmentation**.

**Analogy:**
Like using a **flexible stencil** that bends to follow the object’s outline instead of staying rigid.

---

## Minimal Code Blueprints (PyTorch-style)

**Residual Block**

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)
```

**SE Block**

```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.mean(dim=(2,3))  # Global Avg Pool
        y = self.fc2(F.relu(self.fc1(y)))
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y
```

---

## When to Use What (Practical Tips)

* **Training unstable?** → Add BN (or LN for Transformers).
* **Very deep CNN?** → Add residual connections.
* **Model missing key parts?** → Add SE/CBAM for attention.
* **Segmentation task?** → Try dilated convs.
* **Deformable objects?** → Use deformable convs.

---

## Easy Analogies (For Presentations)

* **Residual =** Shortcut road avoiding traffic signals.
* **Attention =** Spotlight highlighting main performer.
* **Dilated =** Same camera, but wider captured scene.
* **Deformable =** Flexible stencil bending to object.
* **BN/LN =** Normalizing class scores to stay fair.

---
