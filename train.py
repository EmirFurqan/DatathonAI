# Gerekli kütüphanelerin yüklenmesi
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
import os

# Cuda ile çalışıyor mu kontrol etme
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device} kullanılıyor.")

# Veri çoğaltma(Augmentiation) ve normalize etme,tensor çevirme
data_dir = r"C:\Users\Emir Furkan\Desktop\Yeni klasör\sorted_images"
train_transforms = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Veriyi train ve validasyon olarak iki olarak set etme
train_dataset = datasets.ImageFolder(os.path.join(data_dir), transform=train_transforms)
val_dataset = datasets.ImageFolder(os.path.join(data_dir), transform=val_transforms)

# Veri yükleyiciler
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Vision Transformer modeli google vit base kullanılmıştır.(Huggingfaceden bulunmuştur)
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=3
)
model.to(device)


# Overfitting oluşumu gözlenmiştir sonrasında aşağıdaki yöntemler uygulanmışıtr.

# Kayıp fonksiyonu ve optimizer belirleme
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)  # L2 regularization

# Dropout katmanını ekleme
for layer in model.children():
    if isinstance(layer, nn.Dropout):
        layer.p = 0.3  # Dropout oranını %30 olarak ayarlayabilirsiniz.(50 ile de denedik)

# Early Stopping değişkenleri
best_val_loss = float('inf')
patience = 3  # İyileşme görülmeyince erken durdurma yapılacak epoch sayısı
epochs_without_improvement = 0

# Eğitim ve doğrulama
num_epochs = 15 #finetuning olduğundan düşük epoch tercih edildi
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # İleri yayılım
        outputs = model(pixel_values=images).logits
        loss = criterion(outputs, labels)

        # Geri yayılım ve optimizasyon
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # İstatistikler
        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels).item()

    train_accuracy = train_correct / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")#her epoch için gözlemleme verisi yazdırıldı

    # Doğrulama
    model.eval()
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(pixel_values=images).logits
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels).item()

    val_accuracy = val_correct / len(val_loader.dataset)
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

    # Early Stopping Kontrolü(Eğer 3 tane art arda gelişme görülmezse eğitim sona erecek)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        # Modeli kaydetme en iyi val değerli modeli kaydedilmiştir
        torch.save(model.state_dict(), "vit_model_best.pth")
        print("Model en iyi doğrulama kaybı ile kaydedildi.")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping uygulandı. Model doğrulama kaybı {patience} epoch boyunca iyileşmedi.")
            break

# Modeli kaydetme en son eğittiği modeli kaydetmiştir
torch.save(model.state_dict(), "vit_model_final.pth")
print("Model başarıyla kaydedildi!")
