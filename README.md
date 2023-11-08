1.去掉了Transformer里面的encoder，实现了文本生成的框架
2.测试了一下单机多卡的分布式训练：
   torchrun --standalone --nnodes=1 --nproc-per-node=2 myGPT.py
3.参考llama2里面的写法应用了旋转位置编码方式，抛弃了本身attention is all you need论文里面位置编码方式。
4.在双卡分布式训练中测试了两个进程里面最终模型参数是一致的，证明会自动同步梯度更新模型参数
5.在分布式训练中视具体情况来确定是使用模型并行还是数据并行：
	a.在同样的batch_size里面多进程并行训练可以在更少的epoch里面达到类似训练效果，相当于在但卡训练里增大batch_size，节省GPU
	b.DDP速度明显快于DP，后者用的实际是多线程，因为GIL问题，python里面一个进程永远只能执行一个线程。而多核CPU实际可以并行执行多个线程。
6.多卡训练时候模型会多一个属性module，保存或者读取参数的时候要注意,load_state_dict要先处理torch.load进来的字典，或者直接保存Moudle之后的字典。

