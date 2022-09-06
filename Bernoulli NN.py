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

LF = str(input('Lastfall(e -> Einzellast, s -> Streckenlast)'))
if(LF != 'e' and LF != 's'):
    raise Exception('Ungültiger Lastfall')

if (LF == 'e'):
    #Definition der Parameter des statischen Ersatzsystems
    Lp = float(input('Abstand Kraftangriffspunkt - Einspannung [m]:'))
    P  = float(input('Einzellast [kN]:'))
    Lb = float(input('Länge des Kragarms [m]:'))
    EI = float(input('EI des Balkens [10^-6 kNcm²]'))

    #ODE als Loss-Funktion, Einzellast
    def f(x, net):
        u = net(x)  # ,p,px)
        u_x = torch.autograd.grad(u, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        u_xx = torch.autograd.grad(u_x, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        ode = u_xx + P * (Lp - x) / EI * (x <= Lp )
        return ode

elif(LF == 's'):
    #ODE als Loss-Funktion, Streckenlast
    Ln = float(input('Länge Einspannung bis Anfang Streckenlast [m]'))
    Lq = float(input('Länge Streckenlast [m]'))
    q  = float(input('Streckenlast [kN/m]'))
    Lb = float(input('Länge des Kragarms [m]:'))
    EI = float(input('EI des Balkens [10^-6 kNcm²]'))

    def f(x, net):
        u = net(x)  # ,p,px)
        u_x = torch.autograd.grad(u, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        u_xx = torch.autograd.grad(u_x, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
        ode = u_xx + q * (Lq + Ln - x) * ((Ln + Lq - x)/2) / EI * (x <= Ln+Lq )
        return ode
x_bc = x_bc = np.linspace(0,5,500)
iterations = 5000
previous_validation_loss = 99999999.0
for epoch in range(iterations):
    optimizer.zero_grad()  # to make the gradients zero
    x_bc = np.linspace(0, 5, 500)
    #linspace x Vektor Länge dritter Eintrag und Werte 0 und 500 gleichmäßig sortiert, Abstand immer gleich
    p_bc = np.random.uniform(low=0, high=5, size=(500, 1))
    px_bc = np.random.uniform(low=0, high=5, size=(500, 1))

    pt_x_bc = torch.unsqueeze(Variable(torch.from_numpy(x_bc).float(), requires_grad=False).to(device), 1)
    #unsqueeze wegen Kompatibilität
    pt_zero = Variable(torch.from_numpy(np.zeros(1)).float(), requires_grad=False).to(device)

    net_bc_out = net(pt_x_bc)  # ,pt_p_bc,pt_px_bc)
    #Wenn man den Linspace Vektor eingibt, wie sieht die Biegelinie aus?
    e1 = (net_bc_out[0] - net_bc_out[1]) / (pt_x_bc[0] - pt_x_bc[1])
    #Der erste und zweite Eintrag vom Linspace Vektor wird eingesetzt und die Steigung soll 0 sein e1=w'
    e2 = net_bc_out[0]
    #e2=w
    mse_bc = mse_cost_function(e1, pt_zero) + mse_cost_function(e2, pt_zero)

    x_collocation = np.random.uniform(low=0.0, high=5, size=(5000, 1))
    all_zeros = np.zeros((5000, 1))

    pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)

    f_out = f(pt_x_collocation, net)  # ,pt_px_collocation,pt_p_collocation,net)

    mse_f = mse_cost_function(f_out, pt_all_zeros)

    loss = mse_bc + mse_f

    loss.backward()
    optimizer.step()
    with torch.autograd.no_grad():
        if epoch % 10 == 9:
            print(epoch, "Traning Loss:", loss.data)












##
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splrep, splev

plt.rcParams['text.usetex'] = True
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

x = np.linspace(0,Lb,1000)
pt_x = torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=False).to(device), 1)

pt_u_out = net(pt_x)
u_out_cpu = pt_u_out.cpu()
u_out = u_out_cpu.detach()
u_out = u_out.numpy()


u_der = np.gradient(np.squeeze(u_out),x)
bspl = splrep(x,u_der,s=5)
u_der_smooth= splev(x,bspl)
u_der2 = np.gradient(np.squeeze(u_der_smooth),x)

fig = plt.figure()

plt.subplot(3, 2, 1)
plt.xlabel('Meter')
plt.ylabel('$v$ [cm]')
plt.plot(x, u_out)
plt.grid()

plt.subplot(3, 2, 3)
plt.xlabel('Meter')
plt.ylabel('$\phi$ $(10^{-2})$')
plt.plot(x, u_der_smooth)
plt.grid()

plt.subplot(3, 2, 5)
plt.xlabel('Meter')
plt.ylabel('$\kappa$ $(10^{-4})$[1/cm]')
plt.plot(x, u_der2)
plt.grid()

#Analytische Lösung Einzellast
#y1= (-5*(500-x*100)/17000000)*10**4
#y2= (-5*(500*(x*100)-0.5*(x*100)**2)/17000000)*10**2
#y3= -5*(250*(x*100)**2-1/6*(x*100)**3)/17000000
#plt.subplot(3, 2, 2)
#plt.xlabel('Meter')
#plt.ylabel('$v$ [cm]')
#plt.plot(x, y3)
#plt.grid()

#plt.subplot(3, 2, 4)
#plt.xlabel('Meter')
#plt.ylabel('$\phi$ $(10^{-2})$')
#plt.plot(x, y2)
#plt.grid()

#plt.subplot(3, 2, 6)
#plt.xlabel('Meter')
#plt.ylabel('$\kappa$ $(10^{-4})$[1/cm]')
#plt.plot(x, y1)
#plt.grid()

#Analytische Lösung Streckenlast
z1= (-q *((Lq + Ln - x)**2)/EI)/2
z2= (q/(3*EI)*((Ln+Lq-x)**3-(Ln+Lq)**3))/2
z3= (q/(3*EI)*(-1/4*(Ln+Lq-x)**4-(Ln+Lq)**3*x+1/4*(Ln+Lq)**4))/2
plt.subplot(3, 2, 2)
plt.xlabel('Meter')
plt.ylabel('$v$ [cm]')
plt.plot(x, z3)
plt.grid()

plt.subplot(3, 2, 4)
plt.xlabel('Meter')
plt.ylabel('$\phi$ $(10^{-2})$')
plt.plot(x, z2)
plt.grid()

plt.subplot(3, 2, 6)
plt.xlabel('Meter')
plt.ylabel('$\kappa$ $(10^{-4})$[1/cm]')
plt.plot(x, z1)
plt.grid()

plt.show()
##






    
