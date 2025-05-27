from math import pi as PI

import time
import torch
import torch.nn as nn
import numpy as np

from hjb_utilities import hjb_create_batch, HJB_ODE, HJBTrain
from utilities import TestNet

from torchdiffeq import odeint_adjoint

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###########################
### Initial Assumptions ###
###########################

T=1.0
N_t = 10
time_points=torch.tensor([0.0,T]).to(device)
dt = T/N_t
integrator_type = 'rk4'

num_theta = 512
num_x_data = 10
batch_size = num_theta*num_x_data

reg_term = 0.00001 ## Regularization term
learning_rate = 0.0005
schedule_rate = 0.1
decay_step = 100

iter_num = 100
writeStep = 5

## params for the control vector field
dropout = False
weight_decay = 1e-5
theta_width = 1000
theta_depth = 4
theta_activation = 'split'

update_model = True
###########################
### Problem Information ###
###########################

epsilon = 0.2 ## Attached to diffusion
mu = 0.5 ## attached to squared gradient

d=8
width = 50
network_size = 2*d*width+width

dict_name = "hjb_"+theta_activation+"_"+str(d)+".pt"
print(dict_name)

print(d,width,network_size)


###########################
### Build V theta Model ###
###########################

v_net = TestNet(network_size,network_size, theta_width, theta_depth, theta_activation, dropout=dropout ).to(device)

v_dict = {}
v_dict["activation"] = theta_activation
v_dict["u_d"] = d
v_dict["u_width"] = width
v_dict["epsilon"] = epsilon
v_dict['mu'] = mu
try:
    loaded_dict = torch.load(dict_name)
    nn.utils.vector_to_parameters(loaded_dict["weights"].to(device), v_net.parameters())
    print("Model Loaded")
except:
    print("New Model")


startTime = time.time()
loss0 = torch.zeros(1).to(device)
optimizer = torch.optim.Adam(v_net.parameters(),lr=learning_rate,weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=schedule_rate)

neural_net = HJBTrain(d,width,epsilon,mu)
forward_ode = HJB_ODE(v_net,neural_net,None,regularization=reg_term)

for epoch in range(1,iter_num+1):
    v_net.zero_grad()
    ### this samples your theta batch.
    theta_batch = hjb_create_batch(num_theta,d,width,device=device).repeat(1,num_x_data).view(batch_size,network_size)

    ### X sampling strategies
    forward_ode.x = neural_net.sampler.sample(torch.randn((theta_batch.shape[0],d)).to(device),theta_batch)
    y_0 = (loss0,theta_batch,)
    forward_out = odeint_adjoint(
        forward_ode,
        y_0,
        time_points,
        method=integrator_type,
        options=dict(step_size=dt),
        adjoint_params=tuple(v_net.parameters())
        )

    loss = forward_out[0][1]

    if epoch % writeStep == 0:
        print("Loss at Step ", epoch, ": ", loss.item())
        print("Total Time: ", time.time()-startTime, " s")
        print("")

    loss.backward()
    optimizer.step()
    scheduler.step()



v_dict["weights"] = nn.utils.parameters_to_vector(v_net.parameters()).cpu()
v_dict["previous_loss"] = loss.item()
v_dict["theta_width"] = theta_width
v_dict["theta_depth"] = theta_depth
if update_model:
    torch.save(v_dict, dict_name)
    print("Model Saved")
