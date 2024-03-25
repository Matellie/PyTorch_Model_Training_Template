from .model import LinearRegression, LogisticRegression, SimpleNeuralNet

import matplotlib.pyplot as plt
import torch
import os

def save_model(model, save_path, model_name='model.pt'):
    torch.save(model.state_dict(), os.path.join(save_path, model_name))

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    return model

def save_loss_graph(loss_history, save_path, graph_name='loss_graph.png'):
    plt.plot(
        [a for a in range(len(loss_history))],
        loss_history
    )
    plt.yscale('log')
    plt.title('Training loss')
    plt.savefig(os.path.join(save_path, graph_name), bbox_inches='tight')