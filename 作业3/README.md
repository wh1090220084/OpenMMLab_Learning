# 作业：基于 RTMDet 的气球检测

背景：熟悉目标检测和 MMDetection 常用自定义流程。

任务：

基于提供的 notebook，将 cat 数据集换成气球数据集;
按照视频中 notebook 步骤，可视化数据集和标签;
使用MMDetection算法库，训练 RTMDet 气球目标检测算法，可以适当调参，提交测试集评估指标;
用网上下载的任意包括气球的图片进行预测，将预测结果发到群里;
按照视频中 notebook 步骤，对 demo 图片进行特征图可视化和 Box AM 可视化，将结果发到群里
需提交的测试集评估指标（不能低于baseline指标的50%）
目标检测 RTMDet-tiny 模型性能 不低于65 mAP。
数据集
气球数据集可以直接下载https://download.openmmlab.com/mmyolo/data/balloon_dataset.zip

P.S. 同时也欢迎各位选择更复杂的数据集进行训练，如选用同济子豪兄的十类饮料目标检测数据集Drink_28

# 展示图片

```python
# 数据集可视化

import os
import matplotlib.pyplot as plt
from PIL import Image

original_images = []
images = []
texts = []
plt.figure(figsize=(16, 5))

image_paths = [filename for filename in os.listdir('balloon/train')][:8]

for i, filename in enumerate(image_paths):
    name = os.path.splitext(filename)[0]

    image = Image.open('balloon/train/' + filename).convert("RGB")

    plt.subplot(2, 4, i + 1)
    plt.imshow(image)
    plt.title(f"{filename}")
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.show()

```
![在这里插入图片描述](https://img-blog.csdnimg.cn/89788dcaac164bcf9ea3c43f98cfc62a.png)
格式转为COCO

```python
import os.path as osp

import mmcv

from mmengine.fileio import dump, load
from mmengine.utils import track_iter_progress


def convert_balloon_to_coco(ann_file, out_file, image_prefix):
    data_infos = load(ann_file)

    annotations = []
    images = []
    obj_count = 0
    for idx, v in enumerate(track_iter_progress(data_infos.values())):
        filename = v['filename']
        img_path = osp.join(image_prefix, filename)
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(
            dict(id=idx, file_name=filename, height=height, width=width))

        for _, obj in v['regions'].items():
            assert not obj['region_attributes']
            obj = obj['shape_attributes']
            px = obj['all_points_x']
            py = obj['all_points_y']
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            x_min, y_min, x_max, y_max = (min(px), min(py), max(px), max(py))

            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=0,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(x_max - x_min) * (y_max - y_min),
                segmentation=[poly],
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{
            'id': 0,
            'name': 'balloon'
        }])
    dump(coco_format_json, out_file)


if __name__ == '__main__':
    convert_balloon_to_coco(ann_file='balloon/train/via_region_data.json',
                            out_file='balloon/train/annotation_coco.json',
                            image_prefix='balloon/train')
    convert_balloon_to_coco(ann_file='balloon/val/via_region_data.json',
                            out_file='balloon/val/annotation_coco.json',
                            image_prefix='balloon/val')
```
展示COCO格式的数据集

```python
from pycocotools.coco import COCO
import numpy as np
import os.path as osp
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from PIL import Image

def apply_exif_orientation(image):
    _EXIF_ORIENT = 274
    if not hasattr(image, 'getexif'):
        return image

    try:
        exif = image.getexif()
    except Exception:
        exif = None

    if exif is None:
        return image

    orientation = exif.get(_EXIF_ORIENT)

    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)
    if method is not None:
        return image.transpose(method)
    return image


def show_bbox_only(coco, anns, show_label_bbox=True, is_filling=True):
    """Show bounding box of annotations Only."""
    if len(anns) == 0:
        return

    ax = plt.gca()
    ax.set_autoscale_on(False)

    image2color = dict()
    for cat in coco.getCatIds():
        image2color[cat] = (np.random.random((1, 3)) * 0.7 + 0.3).tolist()[0]

    polygons = []
    colors = []

    for ann in anns:
        color = image2color[ann['category_id']]
        bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox']
        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h],
                [bbox_x + bbox_w, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y]]
        polygons.append(Polygon(np.array(poly).reshape((4, 2))))
        colors.append(color)

        if show_label_bbox:
            label_bbox = dict(facecolor=color)
        else:
            label_bbox = None

        ax.text(
            bbox_x,
            bbox_y,
            '%s' % (coco.loadCats(ann['category_id'])[0]['name']),
            color='white',
            bbox=label_bbox)

    if is_filling:
        p = PatchCollection(
            polygons, facecolor=colors, linewidths=0, alpha=0.4)
        ax.add_collection(p)
    p = PatchCollection(
        polygons, facecolor='none', edgecolors=colors, linewidths=2)
    ax.add_collection(p)


coco = COCO('balloon/val/annotation_coco.json')
image_ids = coco.getImgIds()
print(image_ids)
np.random.shuffle(image_ids)

plt.figure(figsize=(16, 5))

# 只可视化 8 张图片
for i in range(8):
    image_data = coco.loadImgs(image_ids[i])[0]
    image_path = osp.join('balloon/val/', image_data['file_name'])
    annotation_ids = coco.getAnnIds(
        imgIds=image_data['id'], catIds=[], iscrowd=0)
    annotations = coco.loadAnns(annotation_ids)

    ax = plt.subplot(2, 4, i + 1)
    image = Image.open(image_path).convert("RGB")

    # 这行代码很关键，否则可能图片和标签对不上
    image = apply_exif_orientation(image)

    ax.imshow(image)

    show_bbox_only(coco, annotations)

    plt.title(f"{image_data['file_name']}")
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.show()

```

![在这里插入图片描述](https://img-blog.csdnimg.cn/ce53b81924b04ede894cf4c95df9c3ca.png)
# 配置文件
rtmdet_tiny_1xb12-40e_cat.py
```python
_base_ = 'configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py'

data_root = '.'

# 非常重要
metainfo = {
    # 类别名，注意 classes 需要是一个 tuple，因此即使是单类，
    # 你应该写成 `cat,` 很多初学者经常会在这犯错
    'classes': ('balloon',),
    'palette': [
        (220, 20, 60),
    ]
}
num_classes = 1

# 训练 40 epoch
max_epochs = 40
# 训练单卡 bs= 12
train_batch_size_per_gpu = 12
# 可以根据自己的电脑修改
train_num_workers = 4

# 验证集 batch size 为 1
val_batch_size_per_gpu = 1
val_num_workers = 2

# RTMDet 训练过程分成 2 个 stage，第二个 stage 会切换数据增强 pipeline
num_epochs_stage2 = 5

# batch 改变了，学习率也要跟着改变， 0.004 是 8卡x32 的学习率
base_lr = 12 * 0.004 / (32*8)

# 采用 COCO 预训练权重
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'  # noqa

model = dict(
    # 考虑到数据集太小，且训练时间很短，我们把 backbone 完全固定
    # 用户自己的数据集可能需要解冻 backbone
    backbone=dict(frozen_stages=4),
    # 不要忘记修改 num_classes
    bbox_head=dict(dict(num_classes=num_classes)))

# 数据集不同，dataset 输入参数也不一样
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    pin_memory=False,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='balloon/train/annotation_coco.json',
        data_prefix=dict(img='balloon/train')))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='balloon/train/annotation_coco.json',
        data_prefix=dict(img='balloon/val/')))

test_dataloader = val_dataloader

# 默认的学习率调度器是 warmup 1000，但是 cat 数据集太小了，需要修改 为 30 iter
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=30),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,  # max_epoch 也改变了
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]
optim_wrapper = dict(optimizer=dict(lr=base_lr))

# 第二 stage 切换 pipeline 的 epoch 时刻也改变了
_base_.custom_hooks[1].switch_epoch = max_epochs - num_epochs_stage2

val_evaluator = dict(ann_file=data_root + 'annotations/test.json')
test_evaluator = val_evaluator

# 一些打印设置修改
default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='auto'),  # 同时保存最好性能权重
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epochs, val_interval=10)

```

# 训练前的可视化验证

```python
from mmdet.registry import DATASETS, VISUALIZERS
from mmengine.config import Config
from mmengine.registry import init_default_scope
import matplotlib.pyplot as plt
import os.path as osp
cfg = Config.fromfile('rtmdet_tiny_1xb12-40e_cat.py')

init_default_scope(cfg.get('default_scope', 'mmdet'))

dataset = DATASETS.build(cfg.train_dataloader.dataset)
visualizer = VISUALIZERS.build(cfg.visualizer)
visualizer.dataset_meta = dataset.metainfo

plt.figure(figsize=(16, 5))

# 只可视化前 8 张图片
for i in range(8):
   item=dataset[i]

   img = item['inputs'].permute(1, 2, 0).numpy()
   data_sample = item['data_samples'].numpy()
   gt_instances = data_sample.gt_instances
   img_path = osp.basename(item['data_samples'].img_path)

   gt_bboxes = gt_instances.get('bboxes', None)
   gt_instances.bboxes = gt_bboxes.tensor
   data_sample.gt_instances = gt_instances

   visualizer.add_datasample(
            osp.basename(img_path),
            img,
            data_sample,
            draw_pred=False,
            show=False)
   drawed_image=visualizer.get_image()

   plt.subplot(2, 4, i+1)
   plt.imshow(drawed_image[..., [2, 1, 0]])
   plt.title(f"{osp.basename(img_path)}")
   plt.xticks([])
   plt.yticks([])

plt.tight_layout()
plt.show()

```

![在这里插入图片描述](https://img-blog.csdnimg.cn/f86885d320f747d79e1b66ee558ce65e.png)

# 运行

```python
 python tools/train.py rtmdet_tiny_1xb12-40e_cat.py

```
结果：
```python
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.741
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.846
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.823
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.496
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.854
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.228
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.784
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.818
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.733
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.892
06/10 05:35:01 - mmengine - INFO - bbox_mAP_copypaste: 0.741 0.846 0.823 0.000 0.496 0.854
06/10 05:35:01 - mmengine - INFO - Epoch(val) [100][13/13]  coco/bbox_mAP: 0.7410  coco/bbox_mAP_50: 0.8460  coco/bbox_mAP_75: 0.8230  coco/bbox_mAP_s: 0.0000  coco/bbox_mAP_m: 0.4960  coco/bbox_mAP_l: 0.8540  data_time: 0.0033  time: 0.0201

```
# 测试
推理代码：

```python
python tools/test.py rtmdet_tiny_1xb12-40e_cat.py work_dirs/rtmdet_tiny_1xb12-40e_cat/best_coco/bbox_mAP_epoch_90.pth
```
测试结果：
```python
DONE (t=0.01s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.745
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.837
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.815
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.472
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.870
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.230
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.782
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.822
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.692
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.911
06/10 14:19:04 - mmengine - INFO - bbox_mAP_copypaste: 0.745 0.837 0.815 0.000 0.472 0.870
06/10 14:19:04 - mmengine - INFO - Epoch(test) [13/13]  coco/bbox_mAP: 0.7450  coco/bbox_mAP_50: 0.8370  coco/bbox_mAP_75: 0.8150  coco/bbox_mAP_s: 0.0000  coco/bbox_mAP_m: 0.4720  coco/bbox_mAP_l: 0.8700  data_time: 0.5673  time: 1.1063

```
# 测试单张图片
 ![请添加图片描述](https://img-blog.csdnimg.cn/b94eefe2a71344929bdcc15cf50492df.jpeg)

```python
python demo/image_demo.py 975.jpg rtmdet_tiny_1xb12-40e_cat.py --weights work_dirs/rtmdet_tiny_1xb12-40e_cat/best_coco/bbox_mAP_epoch_90.pth
```
测试结果：
![请添加图片描述](https://img-blog.csdnimg.cn/574d3959594a41dd9b81ce1b3a7c7fee.jpeg)
# 特征图可视化
安装mmyolo，执行命令：

```python
git clone -b tutorials https://github.com/open-mmlab/mmyolo.git 
cd mmyolo
pip install -e .
```

首先，resize图片，代码如下：

```python
import cv2

img = cv2.imread('../mmdetection-tutorials/975.jpg')
h,w=img.shape[:2]
resized_img = cv2.resize(img, (640, 640))
cv2.imwrite('resized_image.jpg', resized_img)
```
然后再mmyolo的命令行中执行可视化，命令如下：

```python
python demo/featmap_vis_demo.py resized_image.jpg ../mmdetection-tutorials/rtmdet_tiny_1xb12-40e_cat.py ../mmdetection-tutorials/work_dirs/rtmdet_tiny_1xb12-40e_cat/best_coco/bbox_mAP_epoch_90.pth  --target-layers backbone  --channel-reduction squeeze_mean

```

![在这里插入图片描述](https://img-blog.csdnimg.cn/a4f87919a4004d19be5ca18549ef5f5c.png)


# Grad-Based CAM 可视化
先安装Grad-Based CAM库，执行命令：

```python
pip install grad-cam

```
然后再mmyolo，执行命令：

```python
python demo/boxam_vis_demo.py resized_image.jpg ../mmdetection-tutorials/rtmdet_tiny_1xb12-40e_cat.py ../mmdetection-tutorials/work_dirs/rtmdet_tiny_1xb12-40e_cat/best_coco/bbox_mAP_epoch_90.pth  --target-layers  neck.out_convs[2]

```

![在这里插入图片描述](https://img-blog.csdnimg.cn/554b688fe9b446c1aa4119795897f3d1.jpeg)


