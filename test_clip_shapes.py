import torch
import open_clip

model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
model.eval()

# Hook into resblocks and MHSA to print shapes
def hook_fn(module, input, output):
    if isinstance(input, tuple):
        input_shapes = [i.shape if isinstance(i, torch.Tensor) else type(i) for i in input]
    else:
        input_shapes = input.shape
    
    if isinstance(output, tuple):
        output_shapes = [o.shape if isinstance(o, torch.Tensor) else type(o) for o in output]
    else:
        output_shapes = output.shape
        
    print(f"{module.__class__.__name__}: input={input_shapes} -> output={output_shapes}")

model.visual.transformer.resblocks[0].register_forward_hook(hook_fn)
model.visual.transformer.resblocks[0].attn.register_forward_hook(hook_fn)

x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    model.visual(x)
