# MaskRCNN_in_heart_with-_self_attention
## 1. Introduce

這是一個傳統mask_rcnn的實驗程式 我的目標是使用maskrcnn去分割心臟的心室 以左心室、右心室、心肌為目標。

程式的基底是是使用https://github.com/aotumanbiu/Pytorch-Mask-RCNN 去作修改，用於心臟醫療影像的訓練型態。

在程式中加入self_attention去提高分割的效果。 Self attenation (https://arxiv.org/abs/1803.02155)

還有加入一個是feature share的技術，在訓練的途中建立兩個訓練任務，讓不同的訓練部分可以互相分享邊緣資訊。

https://arxiv.org/abs/2101.07905 (Feature Sharing Cooperative Network for Semantic Segmentation)

## 2.DATASET

使用的數據集為ACDC(Automated Cardiac Diagnosis Challenge) https://www.creatis.insa-lyon.fr/Challenge/acdc/

以下為官方說明:

The targeted population for the study is composed of 150 patients divided into 5 subgroups as follows:

30 normal subjects - NOR

30 patients with previous myocardial infarction (ejection fraction of the left ventricle lower than 40% and several myocardial segments with abnormal contraction) - MINF

30 patients with dilated cardiomyopathy (diastolic left ventricular volume >100 mL/m2 and an ejection fraction of the left ventricle lower than 40%) - DCM

30 patients with hypertrophic cardiomyopathy (left ventricular cardiac mass high than 110 g/m2, several myocardial segments with a thickness higher than 15 mm in diastole and a normal ejecetion fraction) - HCM

30 patients with abnormal right ventricle (volume of the right ventricular cavity higher than 110 mL/m2 or ejection fraction of the rigth ventricle lower than 40%) - RV

Each group was clearly defined according to physiological parameter, such as the left or right diastolic volume or ejection fraction, the local contraction of the LV, the LV mass and the maximum thickness of the myocardium. More details can be found on the Classification rules tab.

## 3. MODULE and detail 

###  Bankbone and RGB

我使用resnet101和resnet50去當作maskrcnn的backbone，因為經過實驗之後去比較resnet101resnet50哪個更容易學到模型特徵，還有數據是在使用RGB和GRAY哪一個會更好。

| backbone  (LV)      | IoU     | MAP   |
|----------------|---------|----------  |
| Resnet 50 FPN RGB  | 0.65    | 0.85   |
| Resnet 101 FPN RGB | 0.571   | 0.79   |


| backbone  (LV)      | IoU     | MAP   |
|----------------|---------|----------  |
| Resnet 101 FPN RGB  | 0.459    | 0.64  |
| Resnet 101 FPN GRAY | 0.482    | 0.743 |

| backbone  (LV)      | IoU     | 
|----------------|---------|-----
| Resnet 101 FPN GRAY | 0.71    |--self_attenation

### feature share 

將心肌與左心室的特徵邊作訓練邊作交換，進行中將兩種任務的特徵從backbone裡提出並且除以二再送入後續的maskrcnn任務中。

| backbone  (LV) feature_share     | IoU     | MAP   |
|----------------|---------|----------  |
| Resnet 50 FPN RGB                | 0.40    | 0.54  |
| Resnet 101 FPN RGB               | 0.29    | 0.45  |
| Resnet 101 FPN GRAY              | 0.427   | 0.567 |


