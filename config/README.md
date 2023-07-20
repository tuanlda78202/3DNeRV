# Config Name Style

```python
{training dataset information}_{algorithm name}_{model component names [component1]_[component2]_[...]}_{training settings}_{testing dataset information}
```

The file name is divided to five parts. All parts and components are connected with `_` and words of each part or component should be connected with `-`.

* {`training` `dataset` `information`}: Training dataset names like `uvghd`, `uvg4k`, etc, and input resolutions. For example: `uvg-ready-mp4-720p` means training on `uvg` dataset, specify ready.mp4 and the input shape is `720x1280`.

* {`algorithm` `name`}: The name of the algorithm, such as `videomae`, `videomaev2`, etc.
  

* {`model` `component` `names`}: Names of the components used in the algorithm such as backbone, head, etc. For example, `vits-p16-emd384` means using ViT Small backbone with patch size 16 and output embedding dim 384.
  
* {`training` `settings`}: Information of training settings such as batch size, frame interval, augmentations, loss, learning rate scheduler, and epochs/iterations. For example: `4xb4-ce-linearlr-40K` means using 4-gpus x 4-images-per-gpu, CrossEntropy loss, Linear learning rate scheduler, and train 40K iterations. Some abbreviations:
  * {`gpu` x `batch_per_gpu`}: GPUs and samples per GPU. `bN` indicates N batch size per GPU. E.g. `8xb2` is the short term of 8-gpus x 2-images-per-gpu. And 4xb4 is used by default if not mentioned.
  * {`schedule`}: training schedule, options are `10k`, `20k`, etc. `10k` and `20k` means 10000 iterations and 20000 iterations respectively.
  
* {`testing` `dataset` `information`} (optional): Testing dataset name for models trained on one dataset but tested on another. If not mentioned, it means the model was trained and tested on the same dataset type.

**Example**: `configs/nerv3d/uvg-mp4-ready-720p_3M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-10k_300e.yaml`
