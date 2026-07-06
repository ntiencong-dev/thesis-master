import torch
import os
from transformers import BlipForImageTextRetrieval

print("Loading BLIP-1 model to extract required heads for Kria CPU...")
model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")

os.makedirs("exported_models", exist_ok=True)

print("Saving vision projection head...")
torch.save(model.vision_proj.state_dict(), "exported_models/blip1_vision_proj.pt")

print("Saving text projection head...")
torch.save(model.text_proj.state_dict(), "exported_models/blip1_text_proj.pt")

print("Saving patch embedder...")
torch.save(model.vision_model.embeddings.state_dict(), "exported_models/blip1_patch_embed.pt")

print("Done! Exported 3 files to exported_models/")
