# 毛坯图
![请添加图片描述](https://img-blog.csdnimg.cn/5a5fdc4b44f24e8989df21bae38e4cc6.jpeg)

# 实现代码

```python
from mmagic.apis import MMagicInferencer
import matplotlib.pyplot as plt
sd_inferencer = MMagicInferencer(model_name='stable_diffusion')

import cv2
import numpy as np
import mmcv
from mmengine import Config
from PIL import Image

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

register_all_modules()
cfg = Config.fromfile('configs/controlnet/controlnet-canny.py')
controlnet = MODELS.build(cfg.model).cuda()
control_img = mmcv.imread('11.JPG')
control = cv2.Canny(control_img, 100, 200)
control = control[:, :, None]
control = np.concatenate([control] * 3, axis=2)
control = Image.fromarray(control)
plt.subplot(121)
plt.imshow(control_img)
plt.subplot(122)
plt.imshow(control)
plt.show()
prompt = 'Make this room full of warmth.'
output_dict = controlnet.infer(prompt, control=control)
samples = output_dict['samples']
for idx, sample in enumerate(samples):
    sample.save(f'sample_{idx}.png')
controls = output_dict['controls']
for idx, control in enumerate(controls):
    control.save(f'control_{idx}.png')

plt.subplot(121)
plt.imshow(control_img)
plt.subplot(122)
sample_0 = mmcv.imread('./sample_0.png')
plt.imshow(sample_0)
plt.show()
```
# 边缘检测图
![在这里插入图片描述](https://img-blog.csdnimg.cn/e58d46fa470b49c19dfacb6a5c903c48.png)
# 生成装修效果图

![在这里插入图片描述](https://img-blog.csdnimg.cn/feff644af3a84608b16727d4090964e8.png)
# 最终效果图
![在这里插入图片描述](https://img-blog.csdnimg.cn/168d4ffede294b9eb993606a407313d3.png)
