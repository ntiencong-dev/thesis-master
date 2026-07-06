import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

os.makedirs("exported_models", exist_ok=True)
image_dir = "data/imagenet"
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg'))][:200]

clip_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

blip_transform = transforms.Compose([
    transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

clip_tensors, blip_tensors = [], []
for img_path in image_files:
    img = Image.open(img_path).convert('RGB')
    clip_tensors.append(clip_transform(img).numpy())
    blip_tensors.append(blip_transform(img).numpy())

np.save("exported_models/calibration_frames_clip.npy", np.stack(clip_tensors))
np.save("exported_models/calibration_frames_blip1.npy", np.stack(blip_tensors))
print(" -> Đã lưu các tensor vào thư mục exported_models/")
