### Torch DDP demo
本目录下的代码主要展示了如何使用`PyTorch`的`DDP`(`Distributed Data Parallel`)来进行分布式训练
+ 不使用分布式训练
`python main.py`

+ 使用分布式训练
`python main_ddp.py`
可以修改修改`CUDA_VISIBLE_DEVICES`来指定不同的显卡。