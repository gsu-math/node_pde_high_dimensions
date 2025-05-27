import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


### Takes torch function of form F(x) with Mesh grid X,Y
### returns Z for use in a 3d plot.
def getPlot3D(func,d,X,Y,device,dim1,dim2,*args,**kwargs):
    where_plot = kwargs.get('plane_slice',2)
    zero_plane = kwargs.get('zero_plane', False)
    s = X.shape

    inputdim = d
    graphing_dim1 = dim1
    graphing_dim2 = dim2

    Z = np.zeros(s)
    if zero_plane:
        DT = np.zeros((X.shape[0] ** 2, inputdim))
    else:
        DT = np.ones((X.shape[0] ** 2, inputdim)) * X[s[0] // where_plot][s[1] // where_plot]

    # convert mesh into point vector for which the model can be evaluated
    c = 0
    for i in range(s[0]):
    	for j in range(s[1]):
            DT[c,graphing_dim1] = X[i,j]
            DT[c,graphing_dim2] = Y[i,j]
            c = c+1


    # evaluate model
    DT = torch.from_numpy(DT).float().to(device)
    Ep = func(DT)


    # copy output into plotable format
    c = 0
    for i in range(s[0]):
    	for j in range(s[1]):
            Z[i,j] = Ep[c][0].item()
            c = c+1

    return Z

class FFNet(torch.nn.Module):
    def __init__(self,d,dd,width,depth,activation_func, *args, **kwargs):
        super(FFNet, self).__init__()
        self.use_dropout = kwargs.get('dropout',False)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.2)
        is_mlp = not kwargs.get('mlp', False)
        out_bias = kwargs.get('out_bias',False)
        self.d = d
        # first layer is from input (d = dim) to 1st hidden layer (d = width)
        self.linearIn = nn.Linear(d, width, bias=is_mlp)

        # create hidden layers
        self.linear = nn.ModuleList()
        for _ in range(depth):
            self.linear.append(nn.Linear(width, width,bias=is_mlp))

        # output layer is linear
        self.linearOut = nn.Linear(width, dd,bias = out_bias)
        self.activation = activation_func
    def forward(self, x):
        # compute the 1st layer (from input layer to 1st hidden layer)
        x = self.activation(self.linearIn(x)) # Match dimension
        # compute from i to i+1 layer
        for layer in self.linear:
            if self.use_dropout:
                y = self.dropout(x)
            else:
                y = x
            x_temp = self.activation(layer(y))
            x = x_temp
        # return the output layer
        return self.linearOut(x)
    

class ResNet(torch.nn.Module):
    def __init__(self,d,dd,width,depth,activation_func, *args, **kwargs):
        super(ResNet, self).__init__()
        self.use_dropout = kwargs.get('dropout',False)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.2)
        is_mlp = not kwargs.get('mlp', False)
        out_bias = kwargs.get('out_bias',False)
        self.d = d

        self.linearIn = nn.Linear(d, width, bias=is_mlp)
        self.linear = nn.ModuleList()
        for _ in range(depth):
            self.linear.append(nn.Linear(width, width,bias=is_mlp))
        self.linearOut = nn.Linear(width, dd,bias = out_bias)
        self.activation = activation_func
    def forward(self, x):
        # compute the 1st layer (from input layer to 1st hidden layer)
        x = self.activation(self.linearIn(x)) # Match dimension
        for layer in self.linear:
            if self.use_dropout:
                y = self.dropout(x)
            else:
                y = x
            x_temp =  x + self.activation(layer(y))
            x = x_temp
        # return the output layer
        return self.linearOut(x)

class SplitNet(torch.nn.Module):
    def __init__(self,d,dd,width,depth,**kwargs):
        super(SplitNet, self).__init__()
        self.use_dropout = kwargs.get('dropout',False)
        self.net = ResNet(d,dd,width,depth,F.relu,dropout=self.use_dropout)
        self.net_exp = FFNet(d,d,width,depth,F.relu,dropout=self.use_dropout)
        self.combo_net = FFNet(d,d,width,depth,torch.sigmoid)

    def forward(self, x):
        y1 = self.net(x)
        y2 = self.net_exp(x)*x
        scale = self.combo_net(x)
        return scale*(y1+y2)


class GaussNet(torch.nn.Module):
    def __init__(self,d,dd,width, *args, **kwargs):
        super(GaussNet, self).__init__()
        self.d = d
        self.width = width
        # first layer is from input (d = dim) to 1st hidden layer (d = width)
        self.bias = nn.Parameter(2*np.sqrt(1/width)*torch.randn((width,d))-np.sqrt(1/width))
        self.scales = nn.Parameter(2*np.sqrt(1/width)*torch.randn((width,d))-np.sqrt(1/width))
        self.linearOut = nn.Linear(width, dd,bias = False)
    def forward(self, x):
        num_dim = len(x.shape)

        ### Some resizing tricks here
        view_box = [x.shape[i] for i in range(num_dim)]
        view_box.insert(num_dim-1,1)

        repeat_box = [1 for i in range(num_dim+1)]
        repeat_box[num_dim-1] = self.width

        z = x.view(view_box).repeat(repeat_box) # Match dimension
        z = torch.sum(torch.pow(self.scales*(z-self.bias),2),num_dim)
        z = torch.exp(-0.5*z)
        # return the output layer
        return self.linearOut(z)

class TestNet(torch.nn.Module):
    def __init__(self,d,dd,width,depth,activation,dropout=False):
        super(TestNet, self).__init__()
        self.dd = dd
        use_dropout = dropout
        if activation == "split":
            self.model = SplitNet(d,dd,width,depth,dropout=use_dropout)
        else:
            if 'ff' in activation:
                activation = activation_switch(activation.split('ff')[0])
                self.model = FFNet(d,dd,width,depth,activation,dropout=use_dropout)
            else:
                activation = activation_switch(activation)
                self.model = ResNet(d,dd,width,depth,activation,dropout=use_dropout,out_bias=True)


    def forward(self, x):
        output = self.model(x)
        return output


def activation_switch(name):
    if name == "sin":
        act = torch.sin
    elif name == "cos":
        act = torch.cos
    elif name == "relu":
        act = F.relu
    elif name == "softplus":
        act = F.softplus
    elif name == "elu":
        act = F.elu
    elif name == "tanh":
        act = torch.tanh
    elif name == "sigmoid":
        act = torch.sigmoid
    else:
        raise ValueError("Invalid Activation choice, given: " + name)
    return act