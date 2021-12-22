import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from itertools import cycle
import matplotlib.pyplot as plt
import time
import torchvision
import os
import copy
import ssl
import pandas as pd
from torchvision import datasets, models, transforms
from numpy import savetxt
plt.interactive(False)
plt.ion() 
ssl._create_default_https_context = ssl._create_unverified_context

accuracy_train_plot = []
loss_train_plot = []

accuracy_val_plot = []
loss_val_plot = []


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224), # 244
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'WaDaBa_dataset'

model_select = 'AlexNet'
epochs = 20

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4 if x =='train' else 408,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        print(optimizer_ft.param_groups[0]["lr"])

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                
                optimizer.zero_grad()

               
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

             
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            class_correct = list(0. for i in range(5))
            class_total = list(0. for i in range(5))
            if phase == 'train':
                accuracy_train_plot.append(epoch_acc)
                loss_train_plot.append(epoch_loss)
            if phase == 'val':
                accuracy_val_plot.append(epoch_acc)
                loss_val_plot.append(epoch_loss)
                y_test = labels.to('cpu')

                # n_classes = y_test.shape[1]
                y_score = outputs.to('cpu')
                _, predicted = torch.max(y_score, 1)
                y_score = y_score.numpy()

                c = (predicted == y_test).squeeze()
                for i in range(408):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

                y_test = label_binarize(y_test, np.arange(5))
                fpr, tpr, thresholds = metrics.roc_curve(y_test.ravel(), y_score.ravel())


            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))


            for i in range(5):
                if class_total[i] == 0:
                    continue
                print('%d : %.2f %%' % (
                    i + 1, 100 * class_correct[i] / class_total[i]))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

if model_select == "AlexNet":
    model_ft = models.alexnet(pretrained=True)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)
w = np.array([np.sqrt(1632 / (5 * 400)), np.sqrt(1632 / (5 * 400)), np.sqrt(1632 / (5 * 400)), np.sqrt(1632 / (5 * 400)), np.sqrt(1632 / (5 * 32))])
w = torch.from_numpy(w)
w = w.type(torch.FloatTensor)
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
print(device)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=epochs)

y = []

for i in range(epochs):
    y.append(i + 1)


