# lenet-mnist
只使用numpy搭建lenet网络  
Use numpy only to build lenet network  
代码中定义了conv、avgpool、flatten、fc（full connect）、sigmoid、relu、softmax、loss类，各个类包括前向传播和反向传播。model类将网络堆叠起来，用来训练和预测精度。  
Conv, avgpool, flatten, fc(full connect), sigmoid, relu, softmax and loss are defined in the code, and each class includes forward propagation and backward propagation. The model class stacks networks for training and forecasting accuracy.  
运行一个epoch大概700秒，四个epoch后精度达到95%以上  
It takes about 700 seconds to run one epoch, and the accuracy reaches more than 95% after four epoches.  
  
