
### 1. Input to Neurons

* In image processing: inputs to a neuron are the **pixel values** (or feature maps after convolution layers)—numerical values representing intensities or features extracted from images.
* In general machine learning: inputs can be any **features or dataset values**.
* For YOLOv8, the model input layer takes image pixels (often normalized) as features that feed into neurons.

### 2. Neurons

* A neuron receives multiple inputs, multiplies each by a **weight**, sums them, adds a **bias**, and then applies an **activation function**.
* Activation introduces non-linearity—important for learning complex patterns like those in images.

### 3. Weights

* Weights are **parameters that scale input features**.
* They determine **how strongly an input influences the neuron output**.
* In image processing context, weights can be thought of as filters emphasizing certain pixel patterns or features (like edges, colors).
* Training adjusts weights to identify important features—for example, a specific color or shape relating to damage in a car image.
* Source: H2O.ai, GeeksforGeeks, Ultralytics discussions\[5]\[6]\[7]

### 4. Biases

* Bias is an added constant term that shifts the neuron's activation function.
* Biases help the neuron activate even if all inputs are zero; they allow models to **fit data better by shifting decision boundaries**.
* In image terms, bias can help the model adjust predictions independent of specific pixel values.
* Source: multiple neural network explanations\[7]\[5]

### 5. Hidden Layers

* Hidden layers sit between the input and output layers.
* They transform input features via neurons (weights, biases, activations) into more abstract representations.
* In YOLO, convolutional hidden layers extract increasingly complex features (edges, textures, shapes, object parts).
* Complexity and depth of hidden layers allow the model to learn rich hierarchical features.

---

# Summary of Perspectives

| Aspect        | Machine Learning Viewpoint                          | Image Processing (YOLO) Viewpoint                         |
| ------------- | --------------------------------------------------- | --------------------------------------------------------- |
| Inputs        | Feature values from dataset                         | Image pixel intensity values (or processed features)      |
| Weights       | Parameters scaling feature importance               | Filters highlighting important visual patterns            |
| Bias          | Shifts neuron activation, adjusts decision boundary | Offsets for neuron activation independent of pixel values |
| Activation    | Introduce non-linearity via function (e.g., SiLU)   | Same (SiLU, sigmoid for probability outputs)              |
| Neurons       | Processing units combining weighted inputs + bias   | Same; mathematical units modeling feature detection       |
| Hidden Layers | Feature transformation and abstraction              | Multiple convolutional layers capturing image features    |

Both perspectives are **correct** and complementary. In YOLOv8 detection for car damage using images, your sir's emphasis on image pixels and color importance relates to how weights and biases act on visual features, and your machine learning answer reflects the general neural network functionality.

---

# Example: Simple Neuron Computation

Given image pixels as inputs:

```latex
x_1, x_2, x_3
```

(pixel intensities)

Weights:

```latex
w_1, w_2, w_3
```

Bias:

```latex
b
```

Neuron computes:

```latex
z = w_1 x_1 + w_2 x_2 + w_3 x_3 + b
```

Output after activation (e.g., SiLU):

```latex
y = \text{SiLU}(z) = z \times \sigma(z)
```

Where

```latex
\sigma
```

is the sigmoid function.








## Before that explain ip and op with and without Activation Function and then tell imporatance and types of Activation Functions


# Different types of Activation Functions

### 1. Sigmoid (Logistic Function)

**What:**
Maps any input to a value between 0 and 1 using:

```
σ(x) = 1 / (1 + e^(-x))
```

**Why:**
Useful for binary classification, outputting probabilities.

**Drawbacks:**

* Vanishing gradient problem (very small gradients in deep layers).
* Outputs not zero-centered (all positive), slowing learning.
* Not great for deep or complex networks.

---
##Eexplain about gradient and  Zero-Centered

# 🔹 Zero-Centered (Simple Explanation)

### ✅ What it means

* **Zero-centered:** Outputs around **0** (both +ve and -ve).
* **Not zero-centered:** Outputs only positive (or only negative).

---

### ✅ Example

* **Sigmoid:** Range (0,1) → only +ve → **not zero-centered**.
* **Tanh:** Range (-1,1) → +ve & -ve → **zero-centered**.

---

### ✅ Why it matters

* **Not zero-centered (sigmoid):** Gradients biased → slow, zig-zag learning.
* **Zero-centered (tanh):** Balanced updates → faster & smoother training.

---

### ✅ Analogy

* Swing:

  * **Zero-centered:** Push forward & backward → smooth balance.
  * **Not zero-centered:** Push only one way → imbalance.

---

👉 **In short:**
Zero-centered → balanced learning.
Not zero-centered → biased, slower learning.


### 2. Tanh (Hyperbolic Tangent)

**What:**
Maps input to values between -1 and 1 using:

```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

**Why:**
Zero-centered output makes it better for training than sigmoid in most cases.

**Drawbacks:**

* Still suffers from vanishing gradients.
* Saturates for large inputs (gradients approach zero).

---

### 3. ReLU (Rectified Linear Unit)

**What:**
Outputs zero for negative values, and the input itself for positive values:

```
ReLU(x) = max(0, x)
```

**Why:**

* Fast and simple.
* Solves the vanishing gradient problem for positive values.
* Most popular for hidden layers in modern networks.

**Drawbacks:**

* Can cause “dead neurons” (always output zero if weights are set badly).
* No gradient for negative inputs.

---

### 4. Leaky ReLU

**What:**
A small slope (α ≈ 0.01) for negative inputs instead of zero:

```
Leaky ReLU(x) = x, if x >= 0
                αx, if x < 0
```

**Why:**
Prevents dead neurons by letting negative values through a small slope.

**Drawbacks:**

* Adds a hyperparameter (the leakiness α) to tune.
* Slightly more computationally expensive than ReLU.

---

### 5. Softmax

**What:**
Turns a vector of values into probabilities that sum to one:

```
Softmax(x_i) = e^(x_i) / Σ e^(x_j)
```

**Why:**
Used for multi-class classification (usually final layer), outputting the probability for each class.

**Drawbacks:**

* Only works as a final layer, not for hidden layers.
* Can be sensitive to extremely large/small input values (can cause numerical instability).

---

### 6. Swish (SiLU) *(used in YOLOv8)*

**What:**
Multiplies the input by its sigmoid:

```
Swish(x) = x * σ(x)
```

**Why:**
Smooth, non-monotonic, better gradient flow. Often outperforms ReLU in practice and is used in recent architectures like YOLOv8.

**Drawbacks:**

* Slightly slower than ReLU (multiplications and sigmoid involved).
* Can be harder to implement on certain hardware.

---

## Summary Table

| Function       | Range          | Pros                              | Main Drawbacks                |
|----------------|---------------|-----------------------------------|-------------------------------|
| Sigmoid        | (0, 1)        | Probabilities, simple             | Vanishing gradients           |
| Tanh           | (-1, 1)       | Zero-centered output              | Vanishing gradients           |
| ReLU           | [0, ∞)        | Fast, simple, avoids vanishing    | Dead neurons, not zero-centered|
| Leaky ReLU     | (-∞, ∞)       | Prevents dead neurons             | Tuning parameter              |
| PReLU          | (-∞, ∞)       | Learns best slope for negatives   | More params, risk of overfit  |
| Softmax        | (0, 1)        | Multiclass probabilities          | Only for output, instability  |


***

References: V7 Labs , GeeksforGeeks , Towards Data Science , SuperAnnotate.[1][2][6][8]

[1](https://www.v7labs.com/blog/neural-networks-activation-functions)
[2](https://www.geeksforgeeks.org/machine-learning/activation-functions-neural-networks/)
[3](https://arxiv.org/pdf/2010.09458.pdf)
[4](https://www.aitude.com/comparison-of-sigmoid-tanh-and-relu-activation-functions/)
[5](https://encord.com/blog/activation-functions-neural-networks/)
[6](https://www.superannotate.com/blog/activation-functions-in-neural-networks)
[7](https://alleducationjournal.com/assets/archives/2024/vol9issue3/9025.pdf)
[8](https://towardsdatascience.com/activation-functions-in-neural-networks-how-to-choose-the-right-one-cb20414c04e5/)



***




# YOLOv8 Activation Function
###(Before going to this explain with diagrams about SiLU Activation Function)

- YOLOv8 predominantly uses the **SiLU (Sigmoid Linear Unit)** activation function, also called the **Swish** function.
- SiLU improves gradient flow during training compared to older activations like LeakyReLU.
- In the final prediction layers (objectness and class probabilities), **sigmoid** activation is applied to map outputs between 0 and 1 (probabilities).
- Source: Ultralytics (YOLOv8 GitHub discussions), research papers on YOLOv8 implementation[1][2][3][4]

***

# SiLU Activation Function (Sigmoid Linear Unit)

***

### What is SiLU?

- SiLU, also known as the **Swish** activation function, is defined as:
  
  $$
  \text{SiLU}(x) = x \times \sigma(x) = x \times \frac{1}{1 + e^{-x}}
  $$

  where $$ x $$ is the input to the neuron, and $$ \sigma(x) $$ is the logistic sigmoid function.

- It combines the properties of the **sigmoid** and **linear** functions into one.

***

### How SiLU Works?

- The neuron output is the input multiplied by its sigmoid.

- For **positive inputs**, SiLU behaves almost like the identity function ($$ \text{SiLU}(x) \approx x $$), allowing signals to pass nearly unchanged.

- For **negative inputs**, the output smoothly approaches zero, but unlike ReLU, it does not abruptly zero-out negative values. This allows a small gradient even for negative inputs.

- The function is **smooth and differentiable**, unlike ReLU which has a sharp kink at zero.

***

### Why Use SiLU?

- **Smoothness and non-monotonicity:** SiLU is slightly non-monotonic (the function can dip below zero for small negative values), which may provide richer gradient information during training.

- **Better gradient flow:** SiLU helps mitigate the **vanishing gradient problem** by maintaining gradients even for negative inputs, enabling deeper networks to train better.

- **Improved performance:** Many studies and practical experiments (e.g., in YOLOv8) show SiLU leads to faster convergence and higher accuracy compared to ReLU and sigmoid alone.

- **Self-gating:** The output is gated by the sigmoid of the input, meaning the neuron can modulate its activation adaptively.

***


### Visualization:

Imagine a smooth curve where:

* For large positive `x`, output ≈ `x` (like identity).
* For near-zero `x`, transitions smoothly.
* For large negative `x`, output approaches zero but can be slightly negative (unlike ReLU which clips to zero).

---

### Simple Example Calculation:

Say `x = 2`:

```math
σ(2) = 1 / (1 + e^(-2)) ≈ 0.88
```

```math
SiLU(2) = 2 × 0.88 = 1.76
```

If `x = -1`:

```math
σ(-1) = 1 / (1 + e^(1)) ≈ 0.27
```

```math
SiLU(-1) = -1 × 0.27 = -0.27
```

---


### Where Is SiLU Used?

- In **YOLOv8** (and other modern CNN architectures), SiLU is used to improve feature extraction and detection accuracy.

- Common in networks requiring smooth gradients and better representation capabilities.

***

### References:

- Ultralytics Glossary for YOLOv8: SiLU explanation ()[1]

- PyTorch documentation for `torch.nn.SiLU` ()[6]

- Research Paper: "Sigmoid-weighted linear units for neural network function approximation" ()[7]

- General overview of activation functions (, )[4][5]

***

**In summary**, SiLU is an advanced activation function combining sigmoid gating with input scaling, offering smoothness, better gradient flow, and performance benefits—making it well-suited for complex image detection tasks like those YOLOv8 performs.

[1](https://www.ultralytics.com/glossary/silu-sigmoid-linear-unit)
[2](https://learncplusplus.org/what-is-the-sigmoid-linear-unit-silu-in-a-neural-network-c-app/)
[3](https://www.superannotate.com/blog/activation-functions-in-neural-networks)
[4](https://ml-explained.com/blog/activation-functions-explained)
[5](https://www.slideshare.net/slideshow/7-the-silu-activation-function-unlocking-neural-network-potential-pptx/271381753)
[6](https://docs.pytorch.org/docs/stable/generated/torch.nn.SiLU.html)
[7](https://www.sciencedirect.com/science/article/pii/S0893608017302976)


***

## What is a Gradient in Deep Learning?

- **Gradient** means the mathematical “slope” or “rate of change” of the loss (error) function with respect to each weight in the model.
- During training, the goal of a neural network is to minimize its total error (loss) for prediction.
- Gradients show how much each weight in the network needs to be changed to reduce this error, and in which direction.
- The process of **gradient descent** uses these gradients: at every learning step, weights are updated a little bit using the gradient calculated from comparing the predicted output and the real answer. If the gradient is large, the weight will be changed greatly; if it is small, the weight will be changed just a little.

***

## What is the Vanishing Gradient Problem?

- In deep neural networks with many layers, backpropagation is used to train the model by sending error signals (gradients) from the output layer backwards (to the input layer).
- Sometimes, as these gradients are sent backwards through each layer, they get multiplied by very small numbers—the derivatives (slopes) of activation functions, such as sigmoid or tanh, which are often less than 1 and can be close to zero.
- When you multiply a small number by a small number, the result gets even smaller. In deep networks, after many layers, the gradient can become **extremely tiny, almost zero**—this is the "vanishing" gradient.
- That means the gradients that reach the earlier layers (closer to the input) are so small that the weights in those layers are barely updated at all, **slowing down training or even stopping learning in those layers altogether**.[1][2][5][6]

***

## Simple Example

- Suppose you have a 5-layer network, all using the sigmoid function.
- The derivative (slope) of the sigmoid is always less than 0.25.
- If you backpropagate the error, each layer will multiply the previous gradient by about 0.25.
- So after 5 layers:
  - Gradient = (0.25)^5 = 0.00098 (very tiny).
- If you have 20 layers, the number would be almost zero (vanished), so the weight changes in the first layers are minimal and the network can’t learn deep relationships.

***

## Why Is Vanishing Gradient Bad?

- If the gradient is almost zero, weights barely change and learning stalls in early layers.
- The result: the model fails to learn useful features and overall performance suffers, even if later layers do learn normally.
- The network may get stuck with high error and poor accuracy.

***

## Why Does This Happen?

- Sigmoid and tanh activations "saturate" (output is almost always close to 1 or 0 for sigmoid, -1 or 1 for tanh), so their slope/derivative is close to zero for most input values.
- Chaining many small slopes together shrinks the gradient quickly as you go back through layers.
- The deeper the network, the worse the problem, especially with these activations.

***

## How Can We Fix It?

- Use activation functions with bigger derivatives, like **ReLU** or its variants (Leaky ReLU, ELU).
- Apply **batch normalization** (normalizes layer outputs and keeps gradients steady).
- Use **special architectures** (skip connections found in ResNet help gradients flow better).
- Try **good weight initialization** and, in RNNs, special cells like LSTM or GRU.
- Gradient clipping (set a minimum threshold to keep gradients from getting too small).[2][5][6][1]

***

## Summary

- **Gradient:** tells each weight how much to change to reduce error.
- **Vanishing gradient:** in deep networks, the signal becomes so tiny by the time it reaches early layers that those layers can’t learn.
- This is a major obstacle in deep learning and must be managed to train powerful neural networks.[5][6][1][2]

***

References: GeeksforGeeks, Engati, KDNuggets, DigitalOcean.[6][1][2][5]

[1](https://www.geeksforgeeks.org/deep-learning/vanishing-and-exploding-gradients-problems-in-deep-learning/)
[2](https://www.engati.com/glossary/vanishing-gradient-problem)
[3](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)
[4](https://www.youtube.com/watch?v=8z3DFk4VxRo)
[5](https://www.kdnuggets.com/2022/02/vanishing-gradient-problem.html)
[6](https://www.digitalocean.com/community/tutorials/vanishing-gradient-problem)
[7](https://www.kaggle.com/code/iamvaibhav100/vanishing-gradient)


---

## 🔹 What is an Outlier?

* An **outlier** is a **data point that is very different (much higher or much lower)** compared to most of the other values in the dataset.
* It **doesn’t fit the general pattern** of the data.

👉 In simple words: **“Odd one out” in the dataset.**

---

## 🔹 Examples

1. **Exam Marks**

   * Class scores: 65, 70, 68, 72, 75, 69, 20
   * Here, **20** is an outlier because it’s much lower than others.

2. **Salary Data**

   * Employees’ salaries: ₹40k, ₹45k, ₹50k, ₹55k, ₹3 lakh
   * That ₹3 lakh salary is an **outlier** (too high compared to the rest).

3. **Height of students**

   * Most students: 150–180 cm
   * One student: 220 cm → that’s an **outlier**.

---

## 🔹 Why do outliers matter?

* **Problem in ML/DL models:**

  * Outliers can **distort scaling** (in normalization, min/max gets stretched).
  * They can **pull the mean** too far, making the data look shifted.
  * Some algorithms (like linear regression, K-means) get **heavily affected**.

* **Good side:**

  * Sometimes outliers are **important discoveries**.

    * Example: Detecting fraud in bank transactions (sudden huge amount).
    * Medical data (unusually high blood pressure).

---

## 🔹 How do we handle outliers?

* **Detect them** using:

  * Z-score (>3 or <-3 means outlier).
* **Options to handle:**

  * Remove them (if they are errors).
  * Transform data (log-scaling).
  * Keep them (if they’re meaningful like fraud detection).

---

* **Outlier = unusual data point that’s very far from the rest.**
* Example: One student scoring 20 when everyone else scores 70s.
* They can **spoil scaling and training** but can also be **valuable signals** in cases like fraud detection.
-
***
Got it ✅ You want the explanation of **standardization vs normalization** written in **GitHub Markdown (.md) format** with formulas in `math` blocks and also explained in **very simple words** so you can remember and explain it easily.

Here’s the clean `.md` version for you:

---

# 📊 Standardization vs Normalization

## 🔹 Standardization

* **What it does:** Changes data so that each feature has:

  * **Mean = 0**
  * **Standard deviation = 1**

* **Formula:**

  ```math
  z = (x - mean) / standard deviation
  ```

* **Why use it:**

  * Works well when data looks like a **bell curve (normal distribution)**.
  * Makes features **comparable**, even if they had different scales before.
  * Less affected by extreme values (**outliers**).

* **Result:** Data is **centered around zero** (not fixed to 0–1).
## 🔹 Example 1: Age vs Salary

- Feature 1: Age (20–60)  
- Feature 2: Salary (₹20,000–₹2,00,000)  

❌ Without standardization → salary dominates age (numbers too large).  
✅ With standardization → both features are scaled around 0  
   - Age ~ -1.2 to +1.5  
   - Salary ~ -1.1 to +1.8  

👉 Now the model can compare features on equal footing instead of being biased.

---

## 🔹 Example 2: Weight vs Height (Analogy)

Imagine you’re comparing **weight (in kg)** and **height (in cm)** of students:  

❌ Without standardization → “height” numbers (150–190) look bigger than “weight” (50–100) → model thinks height is more important.  
✅ With standardization → both centered at 0 → model treats them equally and learns real patterns.

---

## 🔹 Normalization

* **What it does:** Scales data to a **fixed range**, usually:

  * `[0, 1]` or sometimes `[-1, 1]`.

* **Formula:**

  ```math
  x_norm = (x - min) / (max - min)
  ```

* **Why use it:**

  * Good when distribution is **unknown** or **not normal**.
  * Makes sure all features contribute **equally** (important for distance-based algorithms like **k-NN, Neural Nets**).
  * Required for algorithms expecting **fixed range inputs**.

* **Result:** All values fit between **0 and 1 (or -1 and 1)**.

* **Example:**
  A feature from 50–500 → after normalization:

  * `50 → 0`
  * `500 → 1`
  * `275 → 0.5`

---

## 🔹 Quick Difference

| Feature           | Standardization                          | Normalization                 |
| ----------------- | ---------------------------------------- | ----------------------------- |
| Output Range      | No fixed range (mean = 0, std = 1)       | Fixed, usually `[0, 1]`       |
| Formula           | `(x - mean) / std`                       | `(x - min) / (max - min)`     |
| Handles Outliers? | ✅ Less sensitive                         | ❌ Very sensitive (min/max)    |
| When to Use?      | Data is normal-like; models like SVM, LR | Data unknown/skewed; NN, k-NN |

---

## 🔹 Easy Way to Remember

* **Standardization = Centering**
  → Think: "Bring all features to a **common standard** around 0".

* **Normalization = Squeezing**
  → Think: "Squeeze all values between **0 and 1**".


# 🧠 Forward Pass & Backpropagation

## 📌 Overview

Neural networks learn through two key steps:

1. **Forward Propagation (Forward Pass):** Input data flows through the network to produce an output.
2. **Backward Propagation (Backpropagation):** The network calculates errors, computes gradients, and updates weights and biases to improve predictions.

---

## 🔁 Forward Propagation (Forward Pass)

Forward propagation is the **prediction phase** of the network — how input features transform into the final output.

---

### 1. **Input Layer**

* Receives raw features $x_1, x_2, ..., x_n$.
* Each input is multiplied by a weight $w$ and added to a bias $b$.

Equation:

$$
z = w_1x_1 + w_2x_2 + ... + w_nx_n + b
$$

---

### 2. **Weighted Sum**

For each neuron in a hidden layer:

$$
z^{[l]} = W^{[l]} \cdot a^{[l-1]} + b^{[l]}
$$

* $W$: Weight matrix
* $b$: Bias vector
* $a^{[l-1]}$: Output from previous layer

---

### 3. **Activation Function**

The weighted sum $z$ is passed through an **activation function** $\sigma(z)$.

* Purpose: Introduce **non-linearity** (so the model learns complex patterns).
* Common activation functions:

  * **Sigmoid:** $\sigma(z) = \frac{1}{1 + e^{-z}}$
  * **ReLU:** $\sigma(z) = \max(0, z)$
  * **Tanh:** $\sigma(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$

Equation:

$$
a^{[l]} = \sigma(z^{[l]})
$$

---

### 4. **Hidden Layers**

* Outputs from one layer become inputs for the next.
* Transformation continues until the final layer.

---

### 5. **Output Layer**

* Produces the prediction $\hat{y}$.
* Example: **Softmax activation** for classification problems.

---

### 🧮 Example Calculation

Suppose:

* Input: $X = [1, 2]$
* Weights: $W = [[0.5, -0.6], [0.1, 0.8]]$
* Biases: $b = [0.2, -0.3]$

Step 1: Weighted sum

$$
z = W \cdot X + b
$$

Step 2: Apply activation (ReLU)

$$
a = \max(0, z)
$$

---

## 🔄 Backward Propagation (Backpropagation)

Backward propagation is the **learning phase** of the network — how it adjusts weights & biases to reduce error.

---

### 1. **Loss Calculation**

Measure error between actual output $y$ and predicted output $\hat{y}$.

Example (Mean Squared Error):

$$
\text{Loss} = \frac{1}{2}(y - \hat{y})^2
$$

---

### 2. **Gradient Calculation**

* Use the **chain rule of calculus** to compute gradients of the loss with respect to each weight and bias.
* Gradient = how much a small change in a parameter affects the loss.

---

### 3. **Weight Update**

Update rule using **gradient descent**:

$$
w = w - \alpha \frac{\partial \text{Loss}}{\partial w}
$$

where $\alpha$ = learning rate.

---

### 4. **Repeat**

* Forward pass → Backward pass → Update weights.
* Repeat for many **epochs** until the loss is minimized.

---

## 🧩 Weights & Biases Explained

* **Weights (w):** Control the strength of connections between neurons.
* **Biases (b):** Allow the model to shift the activation up or down to better fit data.

Equation of a neuron:

$$
y = w \cdot x + b
$$

---

### 🔢 Example of Effect

* If $w = 1, b = 0$: output follows input.
* Changing $w$: changes slope.
* Changing $b$: shifts the line vertically.

---

### 🔁 Role in Training

1. Start with random weights and biases.
2. Forward pass → compute predictions.
3. Compare predictions with actual output.
4. Backward pass → adjust weights & biases using gradients.
5. Repeat until performance improves.

---

## ⚙️ Training Workflow (Step by Step)

1. Initialize weights & biases randomly.
2. **Forward propagation:** Compute predictions.
3. **Loss calculation:** Measure error.
4. **Backward propagation:** Compute gradients.
5. **Update parameters:** Adjust weights & biases.
6. Repeat for multiple iterations (epochs).

---

## 📊 Scale of Parameters in Modern Models

* Small networks → a few thousand parameters.
* Large AI models (like GPT, LLaMA, etc.) → **millions to billions of parameters** (weights & biases).
* These parameters are tuned during training to achieve high accuracy.

---

## ✅ Summary

| Concept              | Role                                                      |
| -------------------- | --------------------------------------------------------- |
| **Forward Pass**     | Data flows through network to produce predictions.        |
| **Backward Pass**    | Gradients are computed, weights & biases updated.         |
| **Weights**          | Control strength of input connections.                    |
| **Biases**           | Shift outputs to fit data better.                         |
| **Activation**       | Introduces non-linearity for learning complex patterns.   |
| **Loss Function**    | Measures difference between predictions & actual results. |
| **Gradient Descent** | Optimization method to minimize loss.                     |

---

## 📘 Notes

* **Forward propagation** is deterministic: same inputs & parameters → same outputs.
* **Backward propagation** is iterative: weights improve step by step.
* Together, they form the **core learning loop** in neural networks.

---

