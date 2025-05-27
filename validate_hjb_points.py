from torchdiffeq import odeint
import matplotlib
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from utilities import GaussNet, TestNet, getPlot3D


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
a=-2
b=2

dict_name = 'hjb_split_8.pt'
fig_save_name = dict_name.split('_')[0] + '_3'

# use_mlp = False
# dict_name = 'AC_(0,1)d2_tanh_2.pt'
theta_dict = torch.load(dict_name, map_location=device)
print(theta_dict["theta_width"], theta_dict["theta_depth"], theta_dict["activation"])

d=theta_dict["u_d"]
width = theta_dict["u_width"]
diffusion = theta_dict['epsilon']
print(d,width)

graphing_points = 50

num_steps = 1000
dt = 1.0/num_steps
t_torch = torch.linspace(0,1.0,num_steps+1).to(device)
t_np = t_torch.cpu().numpy()
print(t_torch.shape)
#
model = GaussNet(d,1,width).to(device)
network_size = nn.utils.parameters_to_vector(model.parameters()).shape[0]
print(network_size)
theta_model = TestNet(network_size,network_size, theta_dict["theta_width"],theta_dict["theta_depth"],theta_dict["activation"]).to(device)
nn.utils.vector_to_parameters(theta_dict["weights"].to(device), theta_model.parameters())
print(theta_dict["previous_loss"], " ", theta_dict["activation"])


def hjb_validate_sample_random(base,d,width,device=torch.device('cpu')):
    # Random
    bias = torch.cat((4*torch.rand(base,width,2)-2,torch.zeros(base,width,d-2)),2).to(device)
    bias = bias.view(base,d*width)
    scales = 1.3*torch.ones((base,d*width),device=device)
    output = -1.0*torch.ones((base,width),device=device)
    output[:,:15]=0.0
    return torch.cat((bias,scales,output),1).detach()

def hjb_validate_sample_triple(base,d,width,device=torch.device('cpu')):
    bias = torch.cat((1*torch.ones(base,width,1),torch.zeros(base,width,d-1)),2).to(device)
    bias[:,30:,0] = -1.0
    bias[:,:15,0] = 0.0
    bias[:,30:,1] = -0.8
    bias[:,15:30,1] = -0.8
    bias[:,:15,1] = 1.0
    bias = bias.view(base,d*width)
    scales = 1.5*torch.ones((base,d*width),device=device)
    output = -1*torch.zeros((base,width),device=device)
    output[:,:15] = -0.8
    output[:,15:30] = -0.8
    output[:,30:] = -0.8
    return torch.cat((bias,scales,output),1).detach()

# theta0 = hjb_validate_sample_random(1,d,width,device=device)[0]
theta0 = hjb_validate_sample_triple(1,d,width,device=device)[0]

def theta_ODE(t,y):
    y_tensor = y
    dydt = theta_model(y_tensor)
    return dydt

with torch.no_grad():
    theta_t = odeint(theta_ODE, theta0, t_torch)

## To run the stochastic process.
num_runs = 300

x_0 = torch.cat((3*torch.rand((graphing_points,2))-1.5,2*torch.rand((graphing_points,d-2))-1),1).to(device)
x_cur = x_0.view(1,graphing_points,d).repeat(num_runs,1,1)
x_t = x_0.view(1,graphing_points,d)
for i,t in enumerate(t_torch[:num_steps]):
    x_cur.requires_grad = True

    nn.utils.vector_to_parameters(theta_t[num_steps-i],model.parameters())
    output = model(x_cur)
    dx, = torch.autograd.grad(output.sum(),x_cur)
    dW = np.sqrt(2*diffusion*dt)*torch.randn((num_runs,graphing_points,d)).to(device)
    x_cur = (x_cur - dt*dx+dW).detach()
    x_t = torch.cat((x_t,x_cur.mean(0).view(1,graphing_points,d)),0).detach()
x_t_final = x_t
print(x_t_final.shape)




fig, axs = plt.subplots(1, 4)
fig.set_size_inches(12,3)
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels

numpoints = 128
# define plotting range and mesh
x = np.linspace(a, b, numpoints)
y = np.linspace(a, b, numpoints)
X, Y = np.meshgrid(x, y)
nn.utils.vector_to_parameters(theta_t[0],model.parameters())
min_level = -15.0
max_level = 0.0
for i in range(4):

    coordinate_1 = 2*i
    coordinate_2 = 2*i+1
    Z_NN = getPlot3D(model,d,X,Y,device,coordinate_1,coordinate_2,zero_plane=True)


    levels = np.linspace(min_level,max_level,15)
    ticks = np.linspace(min_level,max_level,5)


    img = axs[i].contourf(x, y, Z_NN,  levels=levels, cmap='Greys')


    x_1 = x_t_final[:,:,coordinate_1].cpu().detach()
    x_2 = x_t_final[:,:,coordinate_2].cpu().detach()

    i1 = 0
    i2 = (num_steps) // 4
    i3 = (num_steps) // 2
    i4 = (3*num_steps) // 4
    i5 = num_steps
    print(i2,i3,i4)
    axs[i].scatter(x_1[i1].numpy(),x_2[i1].numpy(), marker='o',c='red',s=12)
    axs[i].scatter(x_1[i5].numpy(),x_2[i5].numpy(), marker='^',c='green',s=12)

for ax in axs.flat:
    ax.set(aspect='equal')
    ax.locator_params(nbins=3)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.82, axs[3].get_position().y0, 0.02, axs[0].get_position().y1-axs[0].get_position().y0])
fig.colorbar(img, cax=cbar_ax, ticks = ticks)

for ax in axs.flat:
    ax.label_outer()

fig.savefig("hjb_points_multi5.pdf",bbox_inches="tight")
plt.show()
