# 颜色分类
## dataset  
分为13类，分别为：  
ColorList = ["银", "黑", "绿", "橙", "白", "灰", "红", "蓝", "紫", "黄", "金", "棕", "咖啡"]  
数据集大小约为10000张，按照8：1：1划分训练集，验证集，测试集  
## 分类结果
### 若不对图像进行裁剪：
| 模型 | 准确率 |
| :-----| ----: | :----: |
|VGG19|96|
|mobilenet-v3|91.1|
### 若对图像进行裁剪（只留下目标车辆部分）：
mobilenetv3结果为：93.3
# 朝向分类
## dataset
分为8个朝向，分别为：
DirectList = ['侧前45度车头向右水平', '侧前45度车头向左水平', '侧后45度车头向右水平', '侧后45度车头向左水平', '正侧车头向右水平', '正侧车头向左水平', '正前水平','正后水平']
数据集大小约为10000张，按照8：1：1划分训练集，验证集，测试集
## 分类结果
### 若不对图像进行裁剪：
| 模型 | 准确率 |
| :-----| ----: | :----: |
|VGG19|99.45|
|mobilenet-v3|98.9|
### 若对图像进行裁剪（只留下目标车辆部分）：
mobilenetv3结果为：99.15
# 检测辅助
## dataset
* 整车外观
主要包括：
1.'侧前45度车头向右水平'
2.'侧前45度车头向左水平' 
3.'侧后45度车头向右水平' 
4.'侧后45度车头向左水平'
5.'正侧车头向右水平' 
6.'正侧车头向左水平' 
7.'正前水平'
8.'正后水平'
9.'凡是大于2/3整车的都算作整车'（手动筛选）
* 局部
外观中除整车以外的都为局部
* 内饰
座椅，方向盘，中控，仪表盘，档位，出风口，（引擎盖，后备箱，油盖打开的）都算内饰
数据集约为15000张，三种label的数量约为1：1：1，数据集按照8：1：1划分
## 分类结果
| 模型 | 准确率 |
| :-----| ----: | :----: |
|VGG19|98|
|mobilenet-v3|96.7|