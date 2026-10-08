from transformers import AutoModelForImageClassification
import torch

MODEL_NAME = "mesabo/agri-plant-disease-resnet50"

print("Loading Plant Disease AI model...")

# Load the trained ResNet50 model
model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME
)

model.eval()

print("Model loaded successfully!")
print("Number of classes:", len(model.config.id2label))
print("\nSupported disease classes:")

for index, label in model.config.id2label.items():
    print(index, "->", label)

print("\nPlant Disease Detection AI is ready!")