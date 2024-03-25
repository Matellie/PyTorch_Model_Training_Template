import torch
import torch.nn as nn
import torch.cuda as cuda
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import base_model_code
import base_datasets

import matplotlib.pyplot as plt
import numpy as np
import time
import random

def set_cuda_device():
    if cuda.is_available():
        print(f'Using {cuda.get_device_name(0)}')
        DEVICE = torch.device('cuda:0')
    else:
        print('Using CPU')
        DEVICE = torch.device('cpu')
    return DEVICE

def split_dataset(dataset, val_split, seed=42):
    indices = list(range(len(dataset)))
    split = int(np.floor(val_split * len(dataset)))
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    return train_indices, val_indices

def plot_loss_graph(update_loss_graph, loss_history):
    plt.plot(
        [update_loss_graph*a for a in range(len(loss_history))],
        loss_history
    )
    plt.yscale('log')
    plt.title('Training loss')
    plt.draw()
    plt.pause(0.0001)
    plt.clf()

def show_images(inputs, labels, predictions):
    sqrt = int(np.ceil(np.sqrt(len(inputs))))
    print(f"Show {len(inputs)} images in a {sqrt}x{sqrt} grid")

    figure, axis = plt.subplots(sqrt, sqrt)
    for i in range(sqrt):
        for j in range(sqrt):
            if i*sqrt + j >= len(inputs):
                axis[i, j].axis('off')
            else:
                if labels[i*sqrt + j] == predictions[i*sqrt + j]:
                    color = 'green'
                else:
                    color = 'red'
                image = np.array(inputs[i*sqrt + j].detach().cpu(), dtype='float')
                pixels = image.reshape((8, 8))
                axis[i, j].imshow(pixels, cmap='gray')
                axis[i, j].set_title(f"L:{labels[i*sqrt + j].item()},P:{predictions[i*sqrt + j]}", fontsize=10, color=color)
                axis[i, j].axis('off')
    plt.subplots_adjust(hspace=1.2, wspace=1.2)
    plt.show()

def main():
    device = set_cuda_device()
    batch_size = 1024
    nb_workers = 0
    learning_rate = 0.00001
    nb_epochs = 200000
    update_print = 10000
    update_loss_graph = 1000

    dataset = base_datasets.MNISTDataset()

    # Create data indices for training and validation splits
    train_indices, val_indices = split_dataset(dataset, val_split=0.2, seed=random.randrange(1, 999))
    print(f"Dataset length: {len(dataset)}, Train indices: {len(train_indices)}, Val indices: {len(val_indices)}, Total: {len(train_indices) + len(val_indices)}")
    # Create data samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=nb_workers, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=64, num_workers=nb_workers, sampler=val_sampler)

    # Create model
    model = base_model_code.SimpleNeuralNet(input_size=dataset.get_nb_features(), hidden_size=32, num_classes=10)
    model = model.to(device)

    # Set loss function and optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Set best model and best loss
    best_model = model
    best_loss = torch.tensor(float('inf'))

    # Train model
    loss_history = []
    plt.ion()
    time_train = time.time()
    time_epoch = time.time()
    print(f'{nb_epochs} epochs, print update each {update_print} epochs')
    for epoch in range(nb_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            y_pred = model(inputs)
            l = loss(y_pred, labels)

            l.backward()
            optimizer.step()
            optimizer.zero_grad()

        if (epoch+1) % update_print == 0:
            # Print epoch training time and some infos
            print(f'Epoch {epoch+1}: loss = {l:.4f}, time = {(time.time() - time_epoch):.1f}s')
            time_epoch = time.time()

        if (epoch+1) % update_loss_graph == 0:
            #Save loss history
            loss_history.append(l.detach().cpu().numpy())

            # Plot loss history graph
            plot_loss_graph(update_loss_graph, loss_history)

            # Save best model
            if l.item() < best_loss.item():
                print(f"New best model: {l.item():.4f}")
                best_loss = l
                best_model = model
    print(f'Training time: {(time.time() - time_train):.1f}s')
    plt.ioff()

    # Evaluate best model
    with torch.no_grad():
        correct_guess = torch.tensor(0)
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            y_pred = best_model(inputs)
            y_pred_class = y_pred.argmax(dim=1)
            correct_guess = correct_guess + y_pred_class.eq(labels).sum()

            show_images(inputs, labels, y_pred_class)

        # Compute and print accuracy
        accuracy = correct_guess / float(len(val_indices))
        print(f"Correct guess: {correct_guess}/{len(val_indices)}")
        print(f'Accuracy: {accuracy.item() * 100:.2f}%')

        # Save model and loss graph
        model_id = f'{dataset.get_name()}_{best_model.get_name()}_{accuracy.item() * 100:.0f}'
        model_name =        'model_' +  model_id + '.pt'
        loss_graph_name =   'loss_' +   model_id + '.png'
        save_path = '.'
        base_model_code.save_model(best_model, save_path=save_path, model_name=model_name)
        base_model_code.save_loss_graph(loss_history, save_path=save_path, graph_name=loss_graph_name)

if __name__ == '__main__':
    main()