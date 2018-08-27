# DeepLearning-Framework-for-recsys


框架使用demo见demo.py

example
```python
# 初始化IO管理器
iM = ioManger(batch_size=16)

# 初始化网络
net = Net(iM)

# 添加全连接层
net.add(DotLayer(inputs=['bias_net'], outputs=['aa'], hiddenNum=1))

# 添加sigmoid激活函数
net.add(SigmoidLayer(inputs=['aa'], outputs=['Output']))

# 初始化SGD optimizer
sgd = SGD(lr=0.00003)#,momentum=0.99,nesterov=True)

# 初始化model
model = Model(net=net, optimizer=sgd, lossFunc=cross_entropy_error(), ioManger=iM)

# build model
model.build()

# 开始训练
model.fit(X=X, y=y, validation_split=0.12, epochs=1)
```
