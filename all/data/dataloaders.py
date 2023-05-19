import sys
import os
from torch.utils.data import Subset
from all.data.datasets import VideoDataSet
from all.base.base_dataloader import BaseDataLoader

sys.path.append(os.getcwd())


class VideoDataLoader(BaseDataLoader):
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers=1):
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)

        self.dataset = VideoDataSet()

        # setup dataloader
        full_dataset = VideoDataSet(args)
        full_dataloader = torch.utils.data.DataLoader(
            full_dataset,
            batch_size=args.batchSize,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            sampler=None,
            drop_last=False,
            worker_init_fn=worker_init_fn,
        )
        args.final_size = full_dataset.final_size
        args.full_data_length = len(full_dataset)
        split_num_list = [int(x) for x in args.data_split.split("_")]

        train_ind_list, args.val_ind_list = data_split(
            list(range(args.full_data_length)), split_num_list, args.shuffle_data, 0
        )

        args.dump_vis = args.dump_images or args.dump_videos

        #  Make sure the testing dataset is fixed for every run
        train_dataset = Subset(full_dataset, train_ind_list)
        train_sampler = None
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batchSize,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
            worker_init_fn=worker_init_fn,
        )
