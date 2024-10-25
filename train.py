import numpy as np
import torch
from torch import nn
from sklearn.metrics import precision_score, recall_score, f1_score

def train_epoch(model, loss_fn, optimizer, train_loader):
    train_loss = 0.0
    total_correct = 0
    total_samples = 0
    for input_data, target in train_loader:
        prediction = model(input_data)
        
        loss = loss_fn(prediction, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(prediction, 1)
        total_correct += (predicted == torch.argmax(target, dim=1)).sum().item()
        total_samples += target.size(0)

    train_accuracy = 100 * total_correct / total_samples
    
    return model, train_loss, train_accuracy

def eval_epoch(model, loss_fn, validation_loader):
    validation_loss = 0.0
    total_correct = 0
    total_samples = 0
    for input_data, target in validation_loader:
        prediction = model(input_data)
        
        loss = loss_fn(prediction, target)
        validation_loss += loss.item()
        _, predicted = torch.max(prediction, 1)
        total_correct += (predicted == torch.argmax(target, dim=1)).sum().item()
        total_samples += target.size(0)

    validation_accuracy = 100 * total_correct / total_samples
    return validation_loss, validation_accuracy

def save_model(model, destination_path):
    torch.save(model.state_dict(), destination_path)

def train_model(model, 
                epochs, 
                loss_fn, 
                optimizer, 
                train_loader, 
                validation_loader, 
                destination_path,
                pattience = 0):
    train_losses = []
    validation_losses = []
    minimun_loss = np.inf
    train_accuracies = []
    validation_accuracies = []
    pattience_count = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        model.train()
        model, train_loss, train_accuracy = train_epoch(model, loss_fn, optimizer, train_loader)
        print(f"Train Loss:{train_loss}")
        print(f"Train Accuracy:{train_accuracy}")
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        with torch.no_grad():
            validation_loss, validation_accuracy = eval_epoch(model, loss_fn, validation_loader)
        print(f"Validation Loss:{validation_loss}")
        print(f"Validation Accuracy:{validation_accuracy}")
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)

        if pattience > 0:
            if minimun_loss > validation_loss:
                print(f"Loss Function decreased {minimun_loss} -> {validation_loss}")
                save_model(model, destination_path)
                minimun_loss = validation_loss
                pattience_count = 0
                best_model = model
            else:
                pattience_count += 1
            if pattience_count > pattience:
                return best_model, train_losses, validation_losses, train_accuracies, validation_accuracies
        else:
            if minimun_loss > validation_loss:
                print(f"Loss Function decreased {minimun_loss} -> {validation_loss}")
                save_model(model, destination_path)
                minimun_loss = validation_loss
            else:
                return model, train_losses, validation_losses, train_accuracies, validation_accuracies

    return model, train_losses, validation_losses, train_accuracies, validation_accuracies


