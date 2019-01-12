---
title: Kaggle Human Protein Atlas 蛋白质分类比赛总结和复盘
date: 2019-01-11 15:09:26
category: Kaggle
tags: [中文]
description: 我们的队伍在前两天结束的 Kaggle 竞赛 Human Protein Atlas Image Classification 中获得了最终第九名的成绩，这里做一下简单的复盘和总结。
---

## Intro

这个比赛是典型的 [Multitask + Multilabel](https://scikit-learn.org/stable/modules/multiclass.html#multiclass-and-multilabel-algorithms)，每张图可能包含 0～n 个不同的 label，并且每种 label 的分布差异较大。[赛前必读之一](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/73938)。

数据中包含的图片（经过处理合并后）大概长这个样子：
<img style="max-width:200px; margin-top:10px" src="https://storage.googleapis.com/kaggle-media/competitions/proteins/description_NACC_cropped_opt.png">


## Dataset

训练集 31072 张图，测试 11702 张图。Kaggle 上直接可下载的数据统一为 512 x 512 Grayscale 的图，每张图对应一个合成图（如简介中例子）的一个通道。每一张合成图对应四张图，分别是 R、G、B、Y 通道的灰图。外部数据大约 74000 张包含 4 通道的和未知数量（据说也在 70000 左右）的 label noise 较大的 3 通道图，即没有 Y 通道。我只下载了约 74000 张的 4 通道图，3 通道图没有想到特别好的使用方法。这里值得一提的是在讨论区有很多人提到其中 Y 通道对最终结果几乎没有影响，不过这个对于运算效率来讲影响不大，我并没有去花时间验证这一点。

### External Data

和官方数据不同的是，外部数据即使图片名称上写的是蓝色（`*_blue.jpg`），实际上也是存储为 JPEG 格式的 RGB 图。重点来了，非对应名字的通道仅仅是用来染色的，对于蓝色通道的图来说，只有蓝色通道的是真的数据，也就是一张 grayscale 的图片，而 G 和 R 通道的只是为了辅助显示颜色而存在。

“正确”的方法是，对蓝色的 JPG 图，取出 B 通道，直接存为 grayscale，对于红色的图，取出 R 通道，以此类推。黄色图较为特殊，是由图中的 G 和 R 任意通道表达的，即这两个通道中的数值 *** 应该 *** 是一样的。然后这里就坑爹了，这个 HPA 官方脑子有坑存的是 JPEG，这个有损压缩造成的 G 和 R 数值有偏差，不过我选择只取了 G 通道。阴差阳错的是之后简单实验了一下，选择 R 通道的话效果会变差若干个点，节省了一些时间，也算蒙上的...


### Data Augmentation

一共尝试了两个版本的 augmentation，第二个效果比第一种简单的稍好一些，但是生成每个 batch 的速度稍微慢一点。

第一版：

```python
augment_img = iaa.Sequential([
    iaa.OneOf([
        iaa.Affine(rotate=90),
        iaa.Affine(rotate=180),
        iaa.Affine(rotate=270),
        iaa.Affine(shear=(-30, 30)),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
    ])], random_order=True)
```

第二版（来自队友 Pavel）：

```python
Compose([
    RandomRotate90(),
    Flip(),
    Transpose(),
    OneOf([
        IAAAdditiveGaussianNoise(),
        GaussNoise(),
    ], p=0.2),
    OneOf([
        MotionBlur(p=.2),
        MedianBlur(blur_limit=3, p=.1),
        Blur(blur_limit=3, p=.1),
    ], p=0.2),
    ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=45, p=.5),
    OneOf([
        OpticalDistortion(p=0.3),
        GridDistortion(p=.1),
        IAAPiecewiseAffine(p=0.3),
    ], p=0.2),
    OneOf([
        IAASharpen(),
        IAAEmboss(),
        RandomBrightnessContrast()
    ], p=0.3)
], p=1)
```

两种方法都没有做 Normalize。

### Label Distribution and Oversample

官方训练数据的数据分布如下，另，这个分布没有计算外部数据。其他讨论可以参考上面的链接以及[这里](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/74065)。

```
0 0.2537316
1 0.0246938
2 0.0713048
3 0.0307392
4 0.0365878
5 0.0494860
6 0.0198496
7 0.0555709
8 0.0010437
9 0.0008861
10 0.0005514
11 0.0215234
12 0.0135481
13 0.0105746
14 0.0209917
15 0.0004135
16 0.0104368
17 0.0041353
18 0.0177622
19 0.0291836
20 0.0033870
21 0.0743768
22 0.0157930
23 0.0583868
24 0.0063408
25 0.1620259
26 0.0064590
27 0.0002166
```

28 种不同的 label 分布极为不均，我目前使用了[这里](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/74374#437548)公开的 oversample 的代码，结合后面提到的交叉验证的 ShuffleSplit ，效果显著（提升了大概0.045个点，0.46~0.51+）。

```python
train_df_orig=train_df.copy()    
lows = [15,15,15,8,9,10,8,9,10,8,9,10,17,20,24,26,15,27,15,20,24,17,8,15,27,27,27]
for i in lows:
    target = str(i)
    indicies = train_df_orig.loc[train_df_orig['Target'] == target].index
    train_df = pd.concat([train_df,train_df_orig.loc[indicies]], ignore_index=True)
    indicies = train_df_orig.loc[train_df_orig['Target'].str.startswith(target+" ")].index
    train_df = pd.concat([train_df,train_df_orig.loc[indicies]], ignore_index=True)
    indicies = train_df_orig.loc[train_df_orig['Target'].str.endswith(" "+target)].index
    train_df = pd.concat([train_df,train_df_orig.loc[indicies]], ignore_index=True)
    indicies = train_df_orig.loc[train_df_orig['Target'].str.contains(" "+target+" ")].index
    train_df = pd.concat([train_df,train_df_orig.loc[indicies]], ignore_index=True)
```


### “CV”

和以往不太一样的是我这里除了最后使用的标准的 straitified 5-fold split 之外，在刚开始较长时间内使用的是 "5 boostrap", 即从数据集中 straitified 采样 5 次，测试集占比 25%。

注：代码中的 `MultilabelStratifiedShuffleSplit` 来自[这里](https://github.com/trent-b/iterative-stratification)，Discussion 见[这里](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/67819#latest-438702)。

```python
fold = 0
# 5 bootstrap
mskf = MultilabelStratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=RANDOM_SEED)
# 5 fold 
# mskf = MultilabelStratifiedKFold(n_splits=5, random_state=RANDOM_SEED)
all_files = pd.read_csv('input/train.csv')
# one-hot multilabel, can be replaced with sklearn code...
targets = []
for i in tqdm(range(all_files.shape[0])):  
    y = np.array(list(map(int, all_files.iloc[i].Target.split(' '))))
    y = np.eye(28, dtype=np.float)[y].sum(axis=0)
    targets.append(y)
re = list(mskf.split(all_files.Id, targets))
train_data_list = all_files.iloc[re[fold][0]]
val_data_list = all_files.iloc[re[fold][1]]
```

这里的分割只在官方数据集上进行，然后将外部数据（～74000）全部加入了训练集，获得 ～98000 的训练数据。这个做法其实是有一定风险的：较大可能会 overfit到外部数据。更好的做法是将外部数据也部分分入验证集，尽管后来验证过后发现两种做法没有特别显著的区别。比较有意思的是这样不同的分割方法得分接近的两个提交中有大概 3000 个测试集图片的 label 不完全一样。

```python
external = pd.read_csv('HPAv18RBGY_wodpl.csv')
train_data_list = pd.concat([train_data_list, external], ignore_index=True)
```

这里我做了一件不是特别科学的事情就是我在分割并加入完整外部数据后的训练集上使用了上面的 oversample 的代码，但是使用 oversample 的分布是官方训练数据里的分布。不过使用了这个以后因为成绩提升过于明显，没有验证是否应该只在原数据上做 oversampling。


## Model

我使用的是 `pretrainedmodels` 中的 BN-Inception 和 Xception，还尝试了 SE-ResNeXt50/101 两个相对较深的模型，但是效果并没有太显著的提升。所有模型都使用了在 ImageNet 上 pretrained 的模型。因为 transfer 不同模型代码基本是一样的，所以就贴一个 BN-Inception 的上来，流程是一致的：

1. 替换首层的 3x7x7x64 的 conv 层为 4x7x7x64 （应对 4 通道的图片输入）
    1. 别的参赛者更倾向于从与训练的
2. 把 feature 部分和 classifier 部分衔接的降维改成 global pooling。fast.ai 中习惯使用 global avg pooling 和 global max pooling concat，之前别的比赛中尝试过，区别基本没有所以这次也没有过多的尝试。
3. 将最后一个 linear （全链接）层替换成 BN + Dropout + Linear 这样的设定。
4. 模型输出的是未激活的 logit（因为使用的是 BCEWithLogit 这个 loss function）

模型代码其中 `config` 中的 `num_classes` 是28，其余设置如下，基本是常规的模型。

```python
def get_bninception(config):
    model = bninception(pretrained='imagenet')
    model.global_pool = nn.AdaptiveAvgPool2d(1)
    model.conv1_7x7_s2 = nn.Conv2d(config.channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(1024),
        nn.Dropout(0.3),
        nn.Linear(1024, config.num_classes),
    )
    return model
```

## Training

```python
num_classes = 28
img_width = 256
img_height = 256
channels = 4  # RGBY
lr = 0.001
batch_size = 48
epoch = 30

scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, cooldown=1, min_lr=0.000001)
optimizer = optim.Adam(model.parameters(), lr=config.lr --> 这是 0.001 , weight_decay=1e-6, amsgrad=True)
criterion = nn.BCEWithLogitsLoss()
```

由于是 multilabel 问题，输出 logit 不能再直接 softmax 了，替换成 sigmoid 即可，这里注意的是 sigmoid 之后进行二分的阈值设成 0.5（或者 around 0.5）效果很差，我设置的是 0.15～0.2。训练时计算 F1 Macro 是直接用的固定阈值 0.15

训练过程中我监控 BCE 和 F1 Macro Loss。

```python
THRES = 0.15
loss = criterion(output, target)  # BCE
f1_loss = f1_score(target.cpu().data.numpy(), output.sigmoid().cpu() > THRES, average='macro')  # scikit-learn implementation
```

基本上，训练集的 BCE 会 converge 到 0.68 上下，验证集的 BCE 也会比较接近这个数字。

## Inference

使用完全一样的阈值。这里应该尝试用 OOF 的预测结果做了每个 class 的最优阈值搜索，然并卵。

没有使用任何 TTA。

## Result

由于算力限制，我个人的最好单模（5-fold CV ensemble）在 public LB 是 0.601 左右，另一个队友用我的代码跑了一些更大的模型提升了大概～0.01。

个人的最好成绩是集成了 0.556 (5-fold 256 image BN-Inception) + 0.578（5-fold 512 image BN-Inception）+ 0.575（5-fold 256 image Xception）得到的，更大的图片和模型都交给队友跑了。

其他队友做了一些 stacking，由于我们的 CV 分割不一样所以我没有参与 stacking。

大约完赛两周前用简单平均法进行了集成，之后全队开始摸鱼（说到这个，Pavel 小哥直接出去旅游是最强的哈哈哈），从 public LB 第 4 名一路掉到了第 19，不过看其中有几个超过我们的人的分数变化蠕动，感觉是在 overfit LB。后来 private LB 揭榜以后事实证明确实如此... 我们 shake 到了第 9 名。当然其中运气成分也比较大。。

在地铁上敲出来的初稿，有一些可能不太完善的地方之后修改补充 ：D