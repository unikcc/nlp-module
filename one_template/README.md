###  PyTorch for NLP模板
以文本分类任务为例，实现了一个用于`NLP`任务的`PyTorch`模板，模型为`TextCNN`

#### 特点
+ 简便、灵巧，可以轻松修改配置以适应其他NLP任务

##### 基本用法 
+ `step1` 进入当前目录
+ `step2` 创建环境 & 安装依赖
    + `pip install requirements.txt`
+ `step3` 执行程序
    + `python main.py`

##### 其他用法
+ **使用`nni`自动搜索参数**
  可以运行`sh nni.sh`实现超参数自动搜索