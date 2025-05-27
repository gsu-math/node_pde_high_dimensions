import torch
import torch.nn as nn

from math import pi, sqrt


class HJB_ODE(nn.Module):
    def __init__(self,vector_field,neural_net,x,regularization=0.0):
        super(HJB_ODE,self).__init__()
        self.v_nn = vector_field
        self.neural_net = neural_net
        self.x = x
        self.reg_term = regularization
        
    ### y should be a tuple containing (loss,theta)
    def forward(self,t,y):
        cur_loss, cur_theta = y
        dtheta_dt = self.v_nn(cur_theta)
        dloss = self.running_cost(self.x,cur_theta,dtheta_dt)

        return (dloss,dtheta_dt)

    def running_cost(self,spatial_points,param_points,dtheta_dt):
            theta_gradient = self.neural_net.du_dtheta(spatial_points,param_points)
            operator_out = self.neural_net.hjb_op(spatial_points,param_points)
            ## first du_dtheta term
            projection = torch.sum(theta_gradient*dtheta_dt,1).unsqueeze(-1)
            diff = torch.sum((projection-operator_out)**2,1)
            return (diff + self.reg_term*torch.sum(dtheta_dt**2,1)).mean()
            # return diff/self.gaussian(spatial_points)

    def gaussian(self,x,variance):
        d = x.shape[len(x.shape)-1]
        normalizing_term = (1/sqrt(2*pi*variance))**d
        top = torch.sum((x)**2,1)
        return normalizing_term*torch.exp((-0.5/variance)*top)

class HJBTrain():
    """
    Model implementing the network structure of GaussNet from utilities. 
    This is done to allow computational speedups in the NODE process particularly for computing
    the du_dtheta(x,theta) terms which is challenging to compute using normal pytorch functions.

    Parameters:
     d: Int, the input dimension of the network (output dim is assumed 1)
     width: Int, the hidden width of the network
     epsilon: Float, the diffiusion coefficent to the HJB problem
     mu: Float, the coefficent to the |grad u|^2 term in the HJB

    Tested and compared to regular method.
    Note:
       hjb_op assumes time is now forward time (so no longer reverse)."""
    def __init__(self,d,width,epsilon,mu, *args, **kwargs):
        super(HJBTrain, self).__init__()
        self.d = d
        self.width = width
        self.epsilon = epsilon
        self.mu = mu
        self.du_dtheta = torch.func.grad(lambda x, theta: self.u(x,theta).sum(),argnums=1)
        self.sampler = MixtureSampler(d,width)

    def u(self, x, theta):
        if len(x.shape) == 3:
            t_view1 = [theta.shape[0],1,self.width,self.d]
            t_view2 = [theta.shape[0],1,self.width]
        elif len(x.shape) == 2:
            t_view1 = [theta.shape[0],self.width,self.d]
            t_view2 = [theta.shape[0],self.width]
        else:
            raise ValueError("Only coded for x of shape (N,M,D) or (N,D)")
        bias = theta[:,:self.d*self.width].view(t_view1)
        scales = theta[:,self.d*self.width:2*self.d*self.width].view(t_view1)
        linOut = theta[:,2*self.d*self.width:].view(t_view2)
        num_dim = len(x.shape)

        ### Some resizing tricks here
        view_box = [x.shape[i] for i in range(num_dim)]
        view_box.insert(num_dim-1,1)

        repeat_box = [1 for i in range(num_dim+1)]
        repeat_box[num_dim-1] = self.width

        z = x.view(view_box).repeat(repeat_box) # Match dimension
        z = torch.sum(torch.pow(scales*(z-bias),2),num_dim)
        z = torch.exp(-0.5*z)
        # return the output layer
        return torch.sum(linOut*z,num_dim-1).unsqueeze(-1)
    
    def hjb_op(self,x,theta):
        if len(x.shape) == 3:
            t_view1 = [theta.shape[0],1,self.width,self.d]
            t_view2 = [theta.shape[0],1,self.width]
        elif len(x.shape) == 2:
            t_view1 = [theta.shape[0],self.width,self.d]
            t_view2 = [theta.shape[0],self.width]
        else:
            raise ValueError("Only coded for x of shape (N,M,D) or (N,D)")
        bias = theta[:,:self.d*self.width].view(t_view1)
        scales = theta[:,self.d*self.width:2*self.d*self.width].view(t_view1)
        linOut = theta[:,2*self.d*self.width:].view(t_view2)
        num_dim = len(x.shape)

        ### Some resizing tricks here
        view_box = [x.shape[i] for i in range(num_dim)]
        view_box.insert(num_dim-1,1)

        repeat_box = [1 for i in range(num_dim+1)]
        repeat_box[num_dim-1] = self.width
        z = x.view(view_box).repeat(repeat_box) # Match dimension

        inner = scales*(z-bias)
        e_z = torch.sum(torch.pow(inner,2),num_dim)
        e_z = (linOut*torch.exp(-0.5*e_z)).unsqueeze(-1)

        grad_u = torch.sum(-1.0*scales*inner*e_z,num_dim-1)

        scale_sum = torch.sum(scales**2,num_dim).unsqueeze(-1)
        grad_sum = torch.sum((scales*inner)**2,num_dim).unsqueeze(-1)
        laplace_u = torch.sum(e_z*(grad_sum-scale_sum),num_dim-1)

        return -1.0*self.mu*torch.sum(grad_u**2,num_dim-1).unsqueeze(-1)+self.epsilon*laplace_u

class MixtureSampler():
    def __init__(self,d,width):
        self.d = d
        self.width = width
    def sample(self,z,theta):
        batch = theta.shape[0]
        choose_centers = torch.zeros((batch,self.width,1),device=z.device)
        indexes = torch.randint(self.width,(batch,1),device=z.device)
        choose_centers[torch.arange(choose_centers.size(0)).unsqueeze(1),indexes,0] = 1.0
        num_dim = len(z.shape)
        mean = (theta[:,:self.d*self.width].view(batch,self.width,self.d)*choose_centers).sum(1)
        scales = (theta[:,self.d*self.width:2*self.d*self.width].view(batch,self.width,self.d)*choose_centers).sum(1)
        A = 1/torch.abs(scales)
        y =A*z+mean
        return y.detach()
    def maximize(self,x,theta,model,**kwargs):
        steps = kwargs.get('epoch',200)
        lr = kwargs.get('lr',0.05)
        du_dx = torch.func.grad(lambda x, theta: torch.abs(model(x,theta)).sum(),argnums=0)
        xk = x
        for k in range(steps):
            xk = (xk + lr*du_dx(x,theta)).detach()
        return xk.detach()

def hjb_create_batch(base,d,width,**kwargs):
    device = kwargs.get('device',torch.device('cpu'))
    bias = 4*torch.rand((base,d*width),device=device)-2
    scales = 1.5*torch.rand((base,d*width),device=device)+0.5
    output = -1*torch.rand((base,width),device=device)
    return torch.cat((bias,scales,output),1).detach()
