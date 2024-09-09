from torch.utils.data import Dataset
import logging
from pathlib import Path
import cv2
import torch
from torch import Tensor
from kornia import image_to_tensor
from kornia.color import rgb_to_ycbcr, bgr_to_rgb
def gray_read(img_path) -> Tensor:
    img_n = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img_t = image_to_tensor(img_n).float() / 255
    return img_t
def ycbcr_read(img_path):
    img_n = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img_t = image_to_tensor(img_n).float() / 255
    img_t = rgb_to_ycbcr(bgr_to_rgb(img_t))
    y, cbcr = torch.split(img_t, [1, 2], dim=0)
    return y, cbcr
def dict_to_device(d, device):
    if d is None:
        return None
    for k, v in d.items():
        if isinstance(v, Tensor):
            d[k] = d[k].to(device)
    return d
class TestData(Dataset):
    def __init__(self, root):
        super().__init__()
        root = Path(root)
        self.root = root

        img_list = [x.name for x in sorted(root.glob('ir/*')) if x.suffix in ['.bmp', '.png', '.jpg']]
        logging.info(f'load {len(img_list)} images from {root.name}')
        self.img_list = img_list

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, index: int):
        return self.train_val_item(index)

    def train_val_item(self, index: int):
        name = self.img_list[index]
        ir = gray_read(self.root / 'ir' / name)
        vi, cbcr = ycbcr_read(self.root / 'vi' / name)
        t = torch.cat([ir, vi, cbcr], dim=0)
        ir, vi, cbcr = torch.split(t, [1, 1, 2], dim=0)
        sample = {
            'name': name,
            'ir': ir, 'vi': vi, 'cbcr':cbcr
        }

        return sample


