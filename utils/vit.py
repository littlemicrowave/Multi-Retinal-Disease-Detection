from timm.models import tiny_vit
model = tiny_vit.tiny_vit_21m_384(pretrained=True)

c = 0
for p in model.parameters():
    c += p.numel()

print(c)