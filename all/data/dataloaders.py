from torchvision import transforms
from all.base.base_dataloader import BaseDataLoader
from all.data.datasets import *
from all.data.datasets import ImageProcess
import sys
import os

sys.path.append(os.getcwd())


class KNCDataLoader(BaseDataLoader):
    """Korean Name Card Data Loader (data ~ 82k, data-demo ~ 1.6k)"""

    def __init__(
        self, output_size, crop_size, batch_size, shuffle, validation_split, num_workers
    ):
        self.output_size = output_size
        self.crop_size = crop_size

        image_process = ImageProcess(dir="data_demo")
        self.img_list, self.mask_list = image_process.mask_image_list()

        self.dataset = KNCDataset(
            self.img_list,
            self.mask_list,
            transform=transforms.Compose(
                [
                    Rescale(self.output_size),
                    RandomCrop(self.crop_size),
                    Normalize,
                ]
            ),
        )

        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )
