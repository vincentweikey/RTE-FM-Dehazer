# -----------------------------------------------------------
# 1. Install once:
#    pip install torchinfo
# -----------------------------------------------------------
import torch
import torch.nn as nn
from torchinfo import summary
from collections import OrderedDict
from models.unet import EfficientUNet
# -----------------------------------------------------------
# 2. Helper: pretty tree of all sub-modules
# -----------------------------------------------------------
def print_tree(model, indent=0, max_depth=3):
    """
    Recursively print every nn.Module inside model.
    Stops at max_depth to avoid pages of activations.
    """
    prefix = "  " * indent
    for name, child in model.named_children():
        mod_str = str(child).split("(")[0]          # short type name
        print(f"{prefix}{name}: {mod_str}")
        if indent < max_depth:
            print_tree(child, indent + 1, max_depth)

# -----------------------------------------------------------
# 3. Helper: every parameter with shape + count
# -----------------------------------------------------------
def param_table(model, show_values=False):
    """
    Returns an OrderedDict and prints a table.
    If show_values=True the tensor is also printed (use only tiny models).
    """
    total = 0
    rows = []
    for name, p in model.named_parameters():
        num = p.numel()
        total += num
        rows.append((name, str(list(p.shape)), f"{num:,}", str(p.dtype), str(p.device)))
        if show_values:
            rows.append((" >>> value", str(p.detach().cpu().tolist()), "", "", ""))
    # print nice table
    print("-" * 90)
    print(f"{'name':<50} {'shape':<15} {'#params':<10} {'dtype':<8} {'device':<6}")
    print("-" * 90)
    for r in rows:
        print(f"{r[0]:<50} {r[1]:<15} {r[2]:<10} {r[3]:<8} {r[4]:<6}")
    print("-" * 90)
    print(f"Total parameters: {total:,}  ({total/1e6:.2f} M)")
    return OrderedDict(model.named_parameters())

# -----------------------------------------------------------
# 4. Put your model here
# -----------------------------------------------------------
# from my_files import EfficientUNet   # or whatever you have
model = EfficientUNet(in_channels=4,
        model_channels=128,
        out_channels=4,
        num_res_blocks=2,
        attention_resolutions=[8, 16],
        dropout=0.1,
        channel_mult=[1, 2, 4, 4],
        use_scale_shift_norm=True)

# -----------------------------------------------------------
# 5. Run the three views
# -----------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
x = torch.randn(2, 4, 64, 64).to(device)
t = torch.rand(2).to(device)

print("\n1. ===== torchinfo summary (input flow) =====")
summary(model, input_data=[x, t], depth=4, device=device)

print("\n2. ===== module tree (first 3 levels) =====")
print_tree(model, max_depth=3)

print("\n3. ===== parameter table =====")
param_table(model, show_values=False)   # set True only for toy models