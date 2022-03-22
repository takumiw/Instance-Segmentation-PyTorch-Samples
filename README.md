# Instance Segmentation - PyTorch Samples

## Setup
### Prepare environment
#### 1. Create a virtual environment and activate it
```bash
python -m venv .venv
source .venv/bin/activate.fish  # fish
source .venv/bin/activate  # bash
```

#### 2. Install PyTorch and torchvision
```bash
(.venv) pip install torch==1.10.0 torchvision
```

### Install MMDetection
#### 1. Install mmcv-full
Build MMCV from source on Linux or macOS following official [docs](https://mmcv.readthedocs.io/en/latest/get_started/build.html#build-on-linux-or-macos).

```bash
(.venv) git clone https://github.com/open-mmlab/mmcv.git
(.venv) cd mmcv
(.venv) pip install -r requirements/optional.txt
(.venv) CC=clang CXX=clang++ CFLAGS='-stdlib=libc++' MMCV_WITH_OPS=1 pip install -e .
```

#### 2. Install MMDetection
Clone the repository and then install [MMDetection](https://github.com/open-mmlab/mmdetection):

```bash
(.venv) git clone https://github.com/open-mmlab/mmdetection.git
(.venv) cd mmdetection
(.venv) pip install -r requirements/build.txt
(.venv) pip install -v -e .
(.venv) pip install -r requirements/albu.txt
```

#### 3. Install openmim
```bash
(.venv) pip install openmim
```

### Verification
To verify whether MMDetection is installed correctly, we can run the following sample code to initialize a detector and inference a demo image, but first we need to download config and checkpoint files.

```bash
mim download mmdet --config faster_rcnn_r50_fpn_1x_coco --dest .
```

```python
from mmdet.apis import init_detector, inference_detector

config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
inference_detector(model, 'demo/demo.jpg')
```


# Inference
Sample images are downloaded from [Pexels](https://www.pexels.com/ja-jp/search/cat/).