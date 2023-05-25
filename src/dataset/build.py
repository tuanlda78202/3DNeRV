import os

from .datasets import RawFrameDataset, VideoDataset


def build_dataset(is_train, test_mode, args):
    if is_train:
        mode = "train"
        anno_path = os.path.join(args.data_path, "train.csv")

    elif test_mode:
        mode = "test"
        anno_path = os.path.join(args.data_path, "test.csv")

    else:
        mode = "validation"
        anno_path = os.path.join(args.data_path, "val.csv")

    if args.data_set == "UVG-HD":
        dataset = VideoDataset(
            anno_path=anno_path,
            data_root=args.data_root,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args,
        )

    elif args.data_set == "UVG-4K":
        dataset = VideoDataset(
            anno_path=anno_path,
            data_root=args.data_root,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args,
        )

    elif args.data_set == "Bunny":
        dataset = RawFrameDataset(
            anno_path=anno_path,
            data_root=args.data_root,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            filename_tmpl=args.fname_tmpl,
            start_idx=args.start_idx,
            args=args,
        )

    else:
        raise NotImplementedError("Unsupported Dataset")

    return dataset
