import torch
from torch import nn

def train_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              optimizer: torch.optim.Optimizer):

  #import time

  model.train()
  train_loss, train_acc = 0, 0
  #start_time = 0
  for batch, (X, y) in enumerate(dataloader):
    #print(f"Loading data for batch {batch} was {time.time() - start_time}")
    X, y = X.to(device), y.to(device)

    # FORWARD PASS
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    train_loss += loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    train_acc += (y_pred_class == y).sum().item()/len(y_pred)
    #start_time = time.time()

  # Adjust metrics to get average loss and accuracy per batch
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc

def valid_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):

  model.eval()
  valid_loss, valid_acc = 0, 0

  with torch.inference_mode():
    for batch, (X, y) in enumerate(dataloader):
      X, y = X.to(device), y.to(device)

      # FORWARD PASS
      valid_pred_logits = model(X)
      loss = loss_fn(valid_pred_logits, y)
      valid_loss += loss.item()
      valid_pred_labels = valid_pred_logits.argmax(dim=1)
      valid_acc += ((valid_pred_labels == y).sum().item()/len(valid_pred_labels))

  # Adjust metrics to get average loss and accuracy per batch
  valid_loss = valid_loss / len(dataloader)
  valid_acc = valid_acc / len(dataloader)
  return valid_loss, valid_acc
