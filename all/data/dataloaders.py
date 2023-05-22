import sys
import os
from torch.utils.data import Subset, DataLoader
from all.data.datasets import VideoDataSet
from all.base.base_dataloader import BaseDataLoader

sys.path.append(os.getcwd())


class VideoDataLoader(BaseDataLoader):
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers=1):
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)

        self.dataset = VideoDataSet()


class VideoDataLoader(BaseDataLoader):
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers=2):
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)

        self.dataset = VideoDataSet()

        full_dataset = VideoDataSet(config)

        full_dataloader = DataLoader(
            full_dataset,
            batch_size=config.batchSize,
            shuffle=True,
            num_workers=config.workers,
            pin_memory=True,
            sampler=None,
            drop_last=False,
            worker_init_fn=worker_init_fn,
        )

        config.final_size = full_dataset.final_size

        split_num_list = [int(x) for x in config.data_split.split("_")]

        train_ind_list, config.val_ind_list = data_split(
            list(range(len(full_dataset))), split_num_list, args.shuffle_data, 0
        )

        #  Make sure the testing dataset is fixed for every run
        train_dataset = Subset(full_dataset, train_ind_list)
        train_sampler = None
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batchSize,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
            worker_init_fn=worker_init_fn,
        )
