# 训练

```shell
python train.py --config_file configs/softmax_triplet.yml MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('market1501')" DATASETS.ROOT_DIR "(r'./data')

```

```shell
环境说明：

matplotlib>=3.2.2
numpy>=1.18.5
opencv-python>=4.1.2
Pillow>=7.1.2
PyYAML>=5.3.1
requests>=2.23.0
scipy>=1.4.1
torch>=1.7.0
torchvision>=0.8.1
tqdm>=4.41.0
pytorch-ignite=0.4.11
```

ps:arrow_right:**该训练reid项目与person_search项目是独立的！！**训练完reid后，把训练好的权重放到person_search/weights下，切换到peron_search项目中在去进行reid识别【不然有时候会报can't import xxx】。

参数说明：

--config_file: 配置文件路径，默认configs/softmax_triplet.yml

--weights: pretrained weight path

--neck:  If train with BNNeck, options: **bnneck** or no

--test_neck:  BNNeck to be used for test, before or after BNNneck options: **before** or **after**

--model_name: Name of backbone.

--pretrain_choice: Imagenet

--IF_WITH_CENTER: us center loss, True or False.

:fountain_pen:

配置文件的修改：

(注意：项目中有两个配置文件，一个是config下的defaults.py配置文件，一个是configs下的yml配置文件，**一般配置yml文件即可**，当两个配置文件参数名相同的时候以yml文件为主，这个需要注意一下)

mars数据集转market1501数据集训练

修改相应参数，生成market1501数据集

```bash
python convert_mars_to_market.py
```

**configs文件**:

以**softmax_triplet.yml**为例：

```
SOLVER:
  OPTIMIZER_NAME: 'Adam' # 优化器
  MAX_EPOCHS: 120  # 总epochs
  BASE_LR: 0.00035
  IMS_PER_BATCH: 8  # batch
TEST:
  IMS_PER_BATCH: 4 # test batch
  RE_RANKING: 'no'
  WEIGHT: "path"  # test weight path
  FEAT_NORM: 'yes'
OUTPUT_DIR: "/logs" # model save path
```

```
=> Market1501 loaded
Dataset statistics:
  ----------------------------------------
  subset   | # ids | # images | # cameras
  ----------------------------------------
  train    |   751 |    12936 |         6
  query    |   750 |     3368 |         6
  gallery  |   751 |    15913 |         6
  ----------------------------------------
Loading pretrained ImageNet model......


2023-02-24 21:08:22.121 | INFO     | engine.trainer:log_training_loss:194 - Epoch[1] Iteration[19/1484] Loss: 9.194, Acc: 0.002, Base Lr: 3.82e-05
2023-02-24 21:08:22.315 | INFO     | engine.trainer:log_training_loss:194 - Epoch[1] Iteration[20/1484] Loss: 9.156, Acc: 0.002, Base Lr: 3.82e-05
2023-02-24 21:08:22.537 | INFO     | engine.trainer:log_training_loss:194 - Epoch[1] Iteration[21/1484] Loss: 9.119, Acc: 0.002, Base Lr: 3.82e-05
```

训练完成在logs文件夹生成相应权重等

extract_logs.py提取日志

plots.py画图

CSDN:https://blog.csdn.net/z240626191s/article/details/129221510?spm=1001.2014.3001.5501

#  训练预权重下载：

将 **r50_ibn_2.pth，resnet50-19c8e357.pth**放在yolov5_reid/weights下

链接：https://pan.baidu.com/s/1QYvFE6rDSmxNl4VBNBar-A 
提取码：yypn
