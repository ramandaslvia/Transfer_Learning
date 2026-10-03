"""common.py — preprocessing yang dipakai bersama oleh train.py dan infer.py.

Aturan emas (slide 13): preprocessing saat deployment di robot HARUS identik
dengan saat pelatihan. Karena itu transform validasi/deployment didefinisikan
di satu tempat saja.
"""
from torchvision import transforms

SIZE = 224
MEAN = [0.485, 0.456, 0.406]   # statistik ImageNet
STD = [0.229, 0.224, 0.225]

# Pelatihan: augmentasi sesuai slide 21
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Validasi & deployment: tanpa augmentasi
val_tf = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
