import os
import torch
from torchvision import models, transforms
from PIL import Image

weights = models.MobileNet_V3_Large_Weights.DEFAULT
model = models.mobilenet_v3_large(weights=weights).eval()
categories = weights.meta['categories']
transform = weights.transforms()

# Test strawberry detection with our test logic
url = 'https://images.unsplash.com/photo-1464965911861-746a04b4bca6?w=400&q=80'
import urllib.request
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as resp:
    img = Image.open(resp).convert('RGB')

tensor = transform(img).unsqueeze(0)
with torch.no_grad():
    probs = model(tensor).softmax(1)[0]

top5 = torch.topk(probs, 5)
for idx, p in zip(top5.indices, top5.values):
    print(f'{categories[idx]}: {p.item()*100:.2f}%')
