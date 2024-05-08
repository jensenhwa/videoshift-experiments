import os
import shutil
from pathlib import Path

annotations = Path("/vision/u/jphwa/interactadl/ego_view_actions_resized")
foldernames = list(os.listdir(annotations))
for folder in foldernames:
    files = os.listdir(annotations / folder)
    if len(files) == 1 and os.path.isdir(annotations / folder / files[0]):
        print(annotations / folder / files[0])
        shutil.move(annotations / folder / files[0], annotations / f"{folder}_or_{files[0]}")
        shutil.rmtree(annotations / folder)
