# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 17:20:05 2022

@author: Jonas Peter
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer1 = nn.Linear(1, 5)
        self.hidden_layer2 = nn.Linear(5, 9)
        self.hidden_layer3 = nn.Linear(9, 9)
        self.hidden_layer4 = nn.Linear(9, 9)
        self.hidden_layer5 = nn.Linear(9, 9)
        self.hidden_layer6 = nn.Linear(9, 9)
        self.hidden_layer7 = nn.Linear(9, 5)
        self.output_layer = nn.Linear(5, 1)

    def forward(self, x):  # ,p,px):
        inputs = x  # torch.cat([x,p,px],axis=1) # combined two arrays of 1 columns each to one array of 2 columns
        layer1_out = torch.sigmoid(self.hidden_layer1(inputs))
        layer2_out = torch.sigmoid(self.hidden_layer2(layer1_out))
        layer3_out = torch.sigmoid(self.hidden_layer3(layer2_out))
        layer4_out = torch.sigmoid(self.hidden_layer4(layer3_out))
        layer5_out = torch.sigmoid(self.hidden_layer5(layer4_out))
        layer6_out = torch.sigmoid(self.hidden_layer6(layer5_out))
        layer7_out = torch.sigmoid(self.hidden_layer7(layer6_out))
        output = self.output_layer(layer7_out)  ## For regression, no activation is used in output layer
        return output

#Hyperparameter
learning_rate = 0.01

net = Net()
net = net.to(device)
mse_cost_function = torch.nn.MSELoss()  # Mean squared error
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, verbose=True)
#mean_loss = sum(loss)/len(loss)

def f(x, net):
    u = net(x)  # ,p,px)
    u_x = torch.autograd.grad(u, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
    u_xx = torch.autograd.grad(u_x, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
    ode = u_xx + 5 * (2.50 - x) / 17.50 * (x <= 2.50 )
    return ode
#ode = w'' + P[kN] * (Lp[cm] - x[cm]) / EI[kNcm²]
#P = 5kN
#E= 21000 kN/cm², I= 833,33 cm^4, EI = 17500000 kNcm²

# x_bc = x_bc = np.linspace(0,5,500)
iterations = 6000
previous_validation_loss = 99999999.0
for epoch in range(iterations):
    optimizer.zero_grad()  # to make the gradients zero
    x_bc = np.linspace(0, 5, 500)
    #linspace x Vektor Länge dritter Eintrag und Werte 0 und 500 gleichmäßig sortiert, Abstand immer gleich
    p_bc = np.random.uniform(low=0, high=5, size=(500, 1))
    px_bc = np.random.uniform(low=0, high=5, size=(500, 1))

    pt_x_bc = torch.unsqueeze(Variable(torch.from_numpy(x_bc).float(), requires_grad=False).to(device), 1)
    #unsqueeze wegen Kompatibilität
    #pt_p_bc = Variable(torch.from_numpy(p_bc).float(), requires_grad=False).to(device)
    #pt_px_bc = Variable(torch.from_numpy(px_bc).float(), requires_grad=False).to(device)
    pt_zero = Variable(torch.from_numpy(np.zeros(1)).float(), requires_grad=False).to(device)

    net_bc_out = net(pt_x_bc)  # ,pt_p_bc,pt_px_bc)
    #Wenn man den Linspace Vektor eingibt, wie sieht die Biegelinie aus?
    e1 = (net_bc_out[0] - net_bc_out[1]) / (pt_x_bc[0] - pt_x_bc[1])
    #Der erste und zweite Eintrag vom Linspace Vektor wird eingesetzt und die Steigung soll 0 sein e1=w'
    e2 = net_bc_out[0]
    #e2=w
    mse_bc = mse_cost_function(e1, pt_zero) + mse_cost_function(e2, pt_zero)

    x_collocation = np.random.uniform(low=0.0, high=5, size=(5000, 1))
    #px_collocation = np.random.uniform(low=0.0, high=500, size=(5000, 1))
    #p_collocation = np.random.uniform(low=0, high=1000, size=(5000, 1))
    all_zeros = np.zeros((5000, 1))

    pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
    #pt_px_collocation = Variable(torch.from_numpy(px_collocation).float(), requires_grad=False).to(device)
    #pt_p_collocation = Variable(torch.from_numpy(p_collocation).float(), requires_grad=False).to(device)
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)

    f_out = f(pt_x_collocation, net)  # ,pt_px_collocation,pt_p_collocation,net)

    mse_f = mse_cost_function(f_out, pt_all_zeros)

    loss = mse_bc + mse_f

    loss.backward()
    optimizer.step()
    with torch.autograd.no_grad():
        if epoch % 10 == 9:
            print(epoch, "Traning Loss:", loss.data)










#%%
import matplotlib.pyplot as plt

x = np.linspace(0,5,10000)
pt_x = torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=False).to(device), 1)

pt_u_out = net(pt_x)
u_out_cpu = pt_u_out.cpu()
u_out = u_out_cpu.detach()
u_out = u_out.numpy()
plt.plot(x, u_out)
plt.show()







    
