import torch, torch.nn as nn
import numpy as np
import torchvision.models as models
 
class MedicalCNN(nn.Module):
    def __init__(self, n_cls=2):
        super().__init__()
        bb = models.resnet18(pretrained=False)
        bb.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(512, n_cls))
        self.model = bb
    def forward(self, x): return self.model(x)
 
def gradient_saliency(model, x, target):
    model.eval(); inp = x.clone().requires_grad_(True)
    out = model(inp); model.zero_grad()
    out[0, target].backward()
    return inp.grad.data.abs().squeeze().numpy()
 
def integrated_gradients(model, x, baseline=None, steps=50):
    if baseline is None: baseline = torch.zeros_like(x)
    alphas = torch.linspace(0, 1, steps).view(-1,1,1,1)
    interps = (baseline + alphas*(x-baseline)).requires_grad_(True)
    out = model(interps.view(-1, *x.shape[1:]))
    grads = torch.autograd.grad(out.sum(), interps)[0]
    return ((x-baseline)*grads.mean(0)).squeeze().detach().numpy()
 
def occlusion_sensitivity(model, x, patch_size=16, stride=8):
    orig = model(x).detach()
    h, w = x.shape[2], x.shape[3]
    sensitivity = np.zeros((h, w))
    for i in range(0, h-patch_size, stride):
        for j in range(0, w-patch_size, stride):
            x_occ = x.clone(); x_occ[:, :, i:i+patch_size, j:j+patch_size] = 0
            occ_out = model(x_occ).detach()
            diff = (orig - occ_occ).abs().mean().item()
            sensitivity[i:i+patch_size, j:j+patch_size] += diff
    return sensitivity
 
model = MedicalCNN(2); x = torch.randn(1, 3, 224, 224)
sal = gradient_saliency(model, x, target=1)
ig  = integrated_gradients(model, x)
print(f"Gradient saliency shape: {sal.shape}")
print(f"Integrated Gradients shape: {ig.shape}")
print(f"IG attribution mean: {np.abs(ig).mean():.4f}")
print("XAI methods ready for clinical interpretation.")
