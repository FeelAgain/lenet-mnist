import numpy as np

def normalize_image(images):
    images=images/255.0
    return images

def one_hot_labels(labels):
    tmp=np.zeros([labels.shape[0],10])
    for i in range(labels.shape[0]):
        tmp[i,labels[i]]=1
    return tmp

with np.load('mnist.npz', allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

x_train = normalize_image(x_train)
x_test = normalize_image(x_test)
y_train = one_hot_labels(y_train)
y_test = one_hot_labels(y_test)

def im2col(image, ksize, stride):
    # image is a 3d tensor([channel, height, weight])
    image_col = []
    for i in range(0, image.shape[1] - ksize + 1, stride):
        for j in range(0, image.shape[2] - ksize + 1, stride):
            col = image[:, i:i + ksize, j:j + ksize].reshape([-1])
            image_col.append(col)
    image_col = np.array(image_col)
    return image_col

def k2col(kernal):
    # ksize=[B,C,H,W]
    kernal_col = np.zeros([(kernal.shape[2]**2)*kernal.shape[1],kernal.shape[0]])
    for b in range(kernal.shape[0]):
        kernal_row=np.array([])
        for c in range(kernal.shape[1]):
            kernal_singal = kernal[b, c].reshape([kernal.shape[2] ** 2])
            kernal_row = np.hstack((kernal_row, kernal_singal))
        kernal_col[:, b] = kernal_row
    return kernal_col

def col2im(y_col, im, kernal):
    y = np.zeros([y_col.shape[1], im.shape[1] - kernal.shape[2] + 1,
                                  im.shape[2] - kernal.shape[3] + 1])     
    for i in range(y_col.shape[1]):
        y[i, :, :] = y_col[:, i].reshape(im.shape[1] - kernal.shape[2] + 1,
                                         im.shape[2] - kernal.shape[3] + 1)
    return y

def col2k(k_col, kernal):
    ksize=kernal.shape[2]
    y = np.zeros(kernal.shape)
    for b in range(kernal.shape[0]):
        for c in range(kernal.shape[1]):
            y[b, c] = k_col[c * (ksize ** 2) : (1 + c) * (ksize ** 2), b].reshape([ksize, ksize])
    return y

class conv:

    def __init__(self, b,c,h,w):
        self.kernal = (np.random.rand(b, c, h, w) - 0.5) / 5

    def forward(self, im):
        self.im = im
        self.im_col = im2col(im, 5, 1)
        self.k_col = k2col(self.kernal)
        self.y_col = np.dot(self.im_col, self.k_col)
        self.y = col2im(self.y_col, self.im, self.kernal)

    def backward(self, delta_last, lr):
        delta_last_col = np.zeros([delta_last.shape[1] ** 2, delta_last.shape[0]])
        for c in range(delta_last.shape[0]):
            delta_last_col[:, c] = delta_last[c, :, :].reshape(delta_last.shape[1] ** 2)
        delta_k_col = np.dot(self.im_col.T, delta_last_col)
        
        # pad
        pad=int(self.kernal.shape[2]-1)
        delta_last_pad = np.zeros([delta_last.shape[0], delta_last.shape[1]+2*pad, delta_last.shape[1]+2*pad])
        for c in range(delta_last.shape[0]):
            delta_last_pad[c, pad: pad + delta_last.shape[1], pad: pad + delta_last.shape[1]] = delta_last[c, :, :]
        # rot
        rot_k=np.zeros(self.kernal.shape)
        for b in range(self.kernal.shape[0]):
            for c in range(self.kernal.shape[1]):
                rot_k[b, c, :, :] = np.flipud(np.fliplr(self.kernal[b, c, :, :]))
        rot_k = rot_k.swapaxes(0, 1)
        # delta
        self.delta = np.zeros(self.im.shape)
        delta_last_pad_col = im2col(delta_last_pad, 5, 1)
        rot_k_col = k2col(rot_k)
        delta_col = np.dot(delta_last_pad_col, rot_k_col)
        self.delta = col2im(delta_col, delta_last_pad, rot_k)
        
        self.kernal -= lr * col2k(delta_k_col,self.kernal)

class avgpool:

    def __init__(self, stride):
        self.stride = stride

    def forward(self, im):
        self.shape = im.shape
        new_h = int(im.shape[1] / self.stride)
        new_w = int(im.shape[2] / self.stride)
        self.y = np.zeros([im.shape[0], new_h, new_w])
        for b in range(im.shape[0]):
            for h in range(new_h):
                for w in range(new_w):
                    self.y[b, h, w] = (1 / (self.stride ** 2)) * np.sum(
                        im[b, 2 * h : 2 * h + self.stride, 2 * w : 2 * w + self.stride])

    def backward(self,delta_last):
        self.delta = np.zeros(self.shape)
        for b in range(delta_last.shape[0]):
            for h in range(delta_last.shape[1]):
                for w in range(delta_last.shape[2]):
                    self.delta[b, h * self.stride : h * (self.stride + 1), w * self.stride : w * (self.stride + 1)] = delta_last[b, h, w] / (self.stride ** 2) 

class flatten:

    def __init__(self):
        pass

    def forward(self, im):
        self.shape = im.shape
        self.y = im.reshape(im.size, 1)

    def backward(self,delta_last):
        self.delta = delta_last.reshape(self.shape)

class fc:

    def __init__(self,size):
        self.w = np.random.rand(size[0], size[1])-0.5

    def forward(self, inputs):
        self.inputs = inputs
        self.y = np.dot(self.w.T, self.inputs)

    def backward(self,delta_last,lr):
        delta_w = np.dot(delta_last, self.inputs.T)
        self.delta = np.dot(self.w, delta_last)
        self.w -= lr * delta_w.T

class sigmoid:
    def __init__(self):
        pass
    def forward(self,x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y
    def backward(self):
        return self.y * (1 - self.y)

class relu:
    def __init__(self):
        pass
    def forward(self,x):
        self.y = x * (x > 0)
        return self.y
    def backward(self,delta_last):
        self.delta = (self.y > 0) * delta_last

class softmax:
    def __init__(self):
        pass
    def forward(self, x):
        shift_x = x - np.max(x) 
        self.y = np.exp(shift_x) / np.sum(np.exp(shift_x))
        return self.y
    def backward(self, t):
        # 该反向传播已包括loss的delta
        self.delta = self.y - t
        pass

class loss:
    def __init__(self):
        pass
    def forward(self, x, t):
        delta = 1e-7
        self.x = x
        self.t = t
        self.y = -t * np.log(x + delta)
    def backward(self):
        self.delta = -(self.t / self.x)
        pass
        

class model:

    def __init__(self):

        self.conv1 = conv(6, 1, 5, 5)
        self.relu1 = relu()
        self.avgpool1 = avgpool(2)
        self.conv2 = conv(16, 6, 5, 5)
        self.relu2 = relu()
        self.avgpool2 = avgpool(2)
        self.flat = flatten()
        self.fc1 = fc([256, 128])
        self.relu3 = relu()
        self.fc2 = fc([128, 64])
        self.relu4 = relu()
        self.fc3 = fc([64, 10])
        self.sm = softmax()
        self.e = loss()

    def forward(self, x):
        x = x.reshape([1,28,28])
        
        self.conv1.forward(x)
        self.relu1.forward(self.conv1.y)
        self.avgpool1.forward(self.relu1.y)
        self.conv2.forward(self.avgpool1.y)
        self.relu2.forward(self.conv2.y)
        self.avgpool2.forward(self.relu2.y)
        self.flat.forward(self.avgpool2.y)
        self.fc1.forward(self.flat.y)
        self.relu3.forward(self.fc1.y)
        self.fc2.forward(self.relu3.y)
        self.relu4.forward(self.fc2.y)
        self.fc3.forward(self.relu4.y)
        self.sm.forward(self.fc3.y)

    def backward(self, t, lr):
        t = t.reshape([10, 1])

        self.sm.backward(t)
        self.fc3.backward(self.sm.delta, lr)
        self.relu4.backward(self.fc3.delta)
        self.fc2.backward(self.relu4.delta, lr)
        self.relu3.backward(self.fc2.delta)
        self.fc1.backward(self.relu3.delta, lr)
        self.flat.backward(self.fc1.delta)
        self.avgpool2.backward(self.flat.delta)
        self.conv2.backward(self.avgpool2.delta, lr)
        self.avgpool1.backward(self.conv2.delta)
        self.conv1.backward(self.avgpool1.delta, lr)
        
    def train(self, x_train, y_train, lr, num):
        for i in range(num):#x_train.shape[0]):
            self.forward(x_train[i])
            self.backward(y_train[i], lr)

    def pred(self, x):
        self.forward(x)
        p = np.argmax(self.sm.y)
        return p
        
    def accu(self, x_test, y_test, num):
        right=0
        for i in range(num):#x_test.shape[0]):
            if self.pred(x_test[i]) == np.argmax(y_test[i]):
                right += 1
        a = right / num
        print(a)
        return a

epoch = 4
lr = [1e-3, 5e-4, 2.5e-4, 1e-5]

import time
np.random.seed(0)

last = time.time()
lenet = model()
for i in range(epoch):
    print('epoch', i + 1)
    lenet.train(x_train, y_train, lr[i], 60000)
    lenet.accu(x_test,y_test,200)
print('avgtime:',(time.time() - last) / epoch)

lenet.accu(x_test, y_test, 10000)