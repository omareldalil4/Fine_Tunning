import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def main():
    # تحديد الجهاز (GPU إن كان متاحاً، وإلا CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # تعريف التحويلات المطلوبة (Resize, ToTensor, Normalize)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # تحميل مجموعة بيانات CIFAR-10
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # تحميل نموذج ResNet-18 المدرب مسبقاً
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features

    # تعديل الطبقة النهائية لتتناسب مع عدد التصنيفات في CIFAR-10 (10 تصنيفات)
    model.fc = nn.Linear(num_ftrs, 10)
    model = model.to(device)

    # تعريف دالة الخسارة والمُحسّن
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # حلقة التدريب (مثال بدورتين فقط)
    num_epochs = 2
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # طباعة الخسارة كل 100 خطوة
            if (i + 1) % 100 == 0:
                avg_loss = running_loss / 100
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {avg_loss:.4f}")
                running_loss = 0.0

    print("Finished fine-tuning the model.")

if __name__ == '__main__':
    main()
