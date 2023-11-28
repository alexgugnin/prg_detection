def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
  import os
  """
  Finds the class folder names in a target directory.

  Assumes target directory is in standard image classification format.

  Args:
    directory (str): target directory to load classnames from.

  Returns:
    Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))

  """
  # 1. Get the class names by scanning the target directory
  classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
  if not classes:
    raise FileNotFoundError(f"Couldn't find any classes in {directory}.")

  class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
  return classes, class_to_idx
