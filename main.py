import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import scipy.io
import matplotlib
import numpy as np

matplotlib.use('Agg')

class AlexNet(nn.Module):
    def __init__(self, num_classes=196): #1000
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

class CarsDataset(Dataset):
    """
       Класс набора данных для  Stanford Cars.
       Аннотации должны содержать поля: 'fname' и 'class'.
       Доступ к каждой строке аннотации осуществляется через: аннотации[idx]
       где:
           annotations[idx]['fname'] - имя файла изображения (например, '00001.jpg')
           annotations[idx]['class'] - номер класса (на основе 1)
   """
    def __init__(self, annotations_file, images_dir, transform=None):
        data = scipy.io.loadmat(annotations_file)
        self.annotations = data['annotations']  # shape: (1, N)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return self.annotations.shape[1]

    def __getitem__(self, idx):
        annotation = self.annotations[0, idx]
        fname = annotation['fname'][0]
        fname = str(fname)
        image_path = os.path.join(self.images_dir, fname)
        class_label = int(annotation['class'][0][0]) - 1
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, class_label

class AdaSmooth(Optimizer):
    """
    Реализация оптимизатора AdaSmooth.
    Он основан на оптимизаторе Адама, но добавляет дополнительное сглаживание
    для динамической адаптации темпа обучения.
    """
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, smooth_factor=0.05):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, smooth_factor=smooth_factor)
        super(AdaSmooth, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)  # First moment
                    state['v'] = torch.zeros_like(p.data)  # Second moment
                    state['smooth'] = torch.zeros_like(p.data)  # Smooth term

                m, v, smooth = state['m'], state['v'], state['smooth']
                beta1, beta2, smooth_factor = group['beta1'], group['beta2'], group['smooth_factor']
                lr, epsilon = group['lr'], group['epsilon']

                state['step'] += 1

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                smooth.mul_(1 - smooth_factor).add_(grad, alpha=smooth_factor)

                m_hat = m / (1 - beta1 ** state['step'])
                v_hat = v / (1 - beta2 ** state['step'])
                smooth_hat = smooth / (1 - (1 - smooth_factor) ** state['step'])

                denom = (v_hat.sqrt() + epsilon) * (1 + smooth_hat.abs())
                p.data.addcdiv_(m_hat, denom, value=-lr)

        return loss

if __name__ == '__main__':
    cars_meta = scipy.io.loadmat('cars_meta.mat')
    class_names = [name[0] for name in cars_meta['class_names'][0]]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CarsDataset('cars_train_annos.mat',
                                'cars_train/cars_train',
                                transform=transform)

    test_dataset = CarsDataset('cars_test_annos_withlabels_eval.mat',
                               'cars_test/cars_test',
                               transform=transform)

    # num_workers=0, чтобы не использовать многопроцессную загрузку:
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AlexNet(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_accuracy = 100 * correct / total
        print(f"Эпоха {epoch + 1}, Потери: {epoch_loss:.4f}, Точность: {epoch_accuracy:.2f}%")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Тестовая точность: {test_accuracy:.2f}%")
