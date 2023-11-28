import torch
from torch import nn
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from model.py import PRGClassificator
from train_func import train
from custom_dataset import CustomDataset

if __name__ == '__main__':
    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from pathlib import Path
    from PIL import Image

    # Setup path to data folder
    image_path = Path("...path")
    if image_path.is_dir():
        print(f"{image_path} directory exists.")

    train_dir = image_path / "train"
    valid_dir = image_path / "validation"
    test_dir = image_path / "test"

    #Defining general transformation for all the images
    #So, cause minimal size is 80x80, we should resize all the images to that size
    general_transforms = transforms.Compose([
        transforms.Resize((80, 80)),#More resolution is not suitable cause minimal size on train set is 100x100
        transforms.ToTensor()
    ])

    #Initialising Dataloaders for parallelising computations
    train_data = CustomDataset(targ_dir=train_dir, transform=general_transforms)
    valid_data = CustomDataset(targ_dir=valid_dir, transform=general_transforms)
    test_data = CustomDataset(targ_dir=test_dir, transform=general_transforms)

    BATCH_SIZE = 4

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=BATCH_SIZE,
                                  num_workers=torch.cuda.device_count()*2,
                                  shuffle=True,
                                  pin_memory=True) #should always be True for training on GPU, speeds up the transfer to it

    valid_dataloader = DataLoader(dataset=train_data,
                                  batch_size=1,
                                  num_workers=torch.cuda.device_count()*2,
                                  shuffle=False,
                                  pin_memory=True)

    test_dataloader = DataLoader(dataset=test_data,
                                batch_size=1,
                                num_workers=torch.cuda.device_count()*2,
                                shuffle=False,
                                pin_memory=True)

    #Creating training grid

    num_blocks = [1]#[1, 2, 3]#4 blocks tested, stable 50%
    hidden_units = [10, 50, 100]#[10, 50, 100, 200, 400]
    learning_rates = [1e-3, 1e-4]#[1e-3, 5e-4, 1e-4, 5e-5]
    dropouts = [0.1, 0.2]
    hyperparams_grid = []
    for block in num_blocks:
      for hu in hidden_units:
        for lr in learning_rates:
          for dropout in dropouts:
            hyperparams_grid.append((block, hu, lr, dropout))

    #Performing training
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    NUM_EPOCHS = 30

    grid_results = []
    for params in hyperparams_grid:
      model_results = {}
      num_blocks = params[0]
      hidden_units = params[1]
      lr = params[2]
      dropout = params[3]
      #Model initialisation
      model = PRGClassificator(input_features=3, # number of color channels
                              hidden_units=hidden_units,
                              output_features=len(train_data.classes),
                              num_blocks = num_blocks,
                              dp = dropout).to(device)

      # Choosing Loss function and optimizer
      loss_fn = nn.CrossEntropyLoss()
      optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

      # Training model
      model_results = train(model_name = f"model_hu{hidden_units}_blocks{num_blocks}_lr{lr}_dp{dropout}",
                            model=model,
                            train_dataloader=train_dataloader,
                            valid_dataloader=test_dataloader,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            epochs=NUM_EPOCHS)
      grid_results.append(model_results)
