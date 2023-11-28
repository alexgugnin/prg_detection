from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from side_funcs import find_classes

class CustomDataset(Dataset):
  def __init__(self, targ_dir: str, transform=None) -> None:
    self.paths = list(Path(targ_dir).glob("*/*"))
    self.transform = transform
    self.classes, self.class_to_idx = find_classes(targ_dir)

  def load_image(self, index: int) -> Image.Image:
    '''
    Opens an image via a path and returns it.
    '''
    image_path = self.paths[index]
    return Image.open(image_path)

  def __len__(self) -> int:
    '''
    Returns the total number of samples.
    '''
    return len(self.paths)

  def __getitem__(self, index: int) -> Tuple[Image.Image, int]:#or Tuple[torch.Tensor, int] with appropriate transform
    '''
    Returns one sample of data, data and label (X, y).
    '''
    img = self.load_image(index)
    class_name  = self.paths[index].parent.name
    class_idx = self.class_to_idx[class_name]

    # Transform if necessary
    if self.transform:
      return self.transform(img), class_idx
    else:
      return img, class_idx
