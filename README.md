# DeepLearning-Framework-for-recsys


框架使用demo见demo.py

example
```python
iM = ioManger(batch_size=16)
net = Net(iM)
net.add(DotLayer(inputs=['bias_net'], outputs=['aa'], hiddenNum=1))
net.add(SigmoidLayer(inputs=['aa'], outputs=['Output']))
sgd = SGD(lr=0.00003)#,momentum=0.99,nesterov=True)
model = Model(net=net, optimizer=sgd, lossFunc=cross_entropy_error(), ioManger=iM)
model.build()
model.fit(X=X, y=y, validation_split=0.12, epochs=1)
```
