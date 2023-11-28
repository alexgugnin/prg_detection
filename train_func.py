import train_and_valid_steps

def train(model_name: str,
          model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          valid_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 20):

  min_val_acc = 0.6
  torch.manual_seed(42)

  from tqdm.auto import tqdm

  results = {"model_name": [],
    "train_loss": [],
    "train_acc": [],
    "valid_loss": [],
    "valid_acc": []
  }
  print(model_name)
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(model=model,
                                        dataloader=train_dataloader,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer)
    valid_loss, valid_acc = valid_step(model=model,
        dataloader=valid_dataloader,
        loss_fn=loss_fn)

    print(
      f"Epoch: {epoch+1} | "
      f"train_loss: {train_loss:.4f} | "
      f"train_acc: {train_acc:.4f} | "
      f"valid_loss: {valid_loss:.4f} | "
      f"valid_acc: {valid_acc:.4f}"
    )

    if valid_acc > min_val_acc:
      torch.save(model.state_dict(), model_name)
      print(f"Saved accuracy is {valid_acc}")
      min_val_acc = valid_acc

    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["valid_loss"].append(valid_loss)
    results["valid_acc"].append(valid_acc)

  results["model_name"].append(model_name)
  return results
