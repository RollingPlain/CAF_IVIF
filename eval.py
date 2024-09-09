import cv2
import tqdm
import torch
import argparse
import yaml
import torch.utils
import os
import sys
from pathlib import Path
from kornia.utils import tensor_to_image
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import from_dict
from model.fuse import Fuse
import logging
from kornia.color import ycbcr_to_rgb, rgb_to_bgr
from datasets.dataset import TestData, dict_to_device
from model.Getcolor_many import Get_color
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
torch.autograd.set_detect_anomaly(True)
class Test:
    def __init__(self, config):
        log_f = '%(asctime)s | %(filename)s[line:%(lineno)d] | %(levelname)s | %(message)s'
        logging.basicConfig(level='INFO', format=log_f)

        if isinstance(config, str) or isinstance(config, Path):
            config = yaml.safe_load(Path(config).open('r', encoding='utf-8'))
            config = from_dict(config)
        else:
            config = config
        self.config = config

        path1 = args.data_path
        t_dataset = TestData(root=path1)

        self.vis_dir = os.path.join(path1, 'vi')
        self.t_loader = DataLoader(t_dataset, batch_size=1, pin_memory=True, num_workers=config.train.num_workers)
        fuse = Fuse(config, mode='val')
        self.fuse = fuse

        checkpoint = torch.load(args.model_path, weights_only=False)
        self.fuse.network.load_state_dict(checkpoint['fuse'].state_dict())

    def eval(self, data_path):
        torch.backends.cudnn.benchmark = True
        process = tqdm(enumerate(self.t_loader),total=len(self.t_loader))

        for idx, sample in process:

            sample = dict_to_device(sample, self.fuse.device)
            with torch.no_grad():

                fus = self.fuse.eval(ir=sample['ir'], vi=sample['vi'])
                if args.color:
                    fus = torch.cat([fus, sample['cbcr']], dim=1)
                    fus = rgb_to_bgr(ycbcr_to_rgb(fus))

                fus_image = tensor_to_image(fus[0] * 255.)

                vis_image_name = sample['name'][0]
                vis_image_path = os.path.join(self.vis_dir, vis_image_name)
                vis_image = cv2.imread(vis_image_path)

                color_image = Get_color(vis_image, fus_image)

            name1 = sample['name'][0]

            cv2.imwrite(str(data_path / name1), color_image)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='cache/Det_final.pth', help='path of model')
    parser.add_argument('--result_path', type=str, default='result', help='path to save the result')
    parser.add_argument('--data_path', type=str, default='data', help='path to datasets')
    parser.add_argument('--cfg', default='config/default.yaml', help='config file path')
    parser.add_argument('--color', action='store_true')

    args = parser.parse_args()
    test = Test(args.cfg)
    test.eval(Path(args.result_path))
