
# 🧠 Fine-Tuning a Neural Network – Class Notes

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
