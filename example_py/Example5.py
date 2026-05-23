# Converted from example/Example5.ipynb
# Cell markers (# %%) are kept for IDE/debugger navigation.

# %%
'''
TORCWA Example5
Shape derivative - Maximize anisotropy

'''
# Import
import numpy as np
import torch
import scipy.io
from matplotlib import pyplot as plt
import time

import torcwa
import Materials

# Hardware
# If GPU support TF32 tensor core, the matmul operation is faster than FP32 but with less precision.
# If you need accurate operation, you have to disable the flag below.
torch.backends.cuda.matmul.allow_tf32 = False
sim_dtype = torch.complex64
geo_dtype = torch.float32
device = torch.device('cuda')

# Simulation environment
# light
lamb0 = torch.tensor(532.,dtype=geo_dtype,device=device)    # nm
inc_ang = 0.*(np.pi/180)    # radian
azi_ang = 0.*(np.pi/180)    # radian

# material
substrate_eps = 1.46**2
silicon_eps = Materials.aSiH.apply(lamb0)**2

# geometry
L = [300., 300.]            # nm / nm
torcwa.rcwa_geo.dtype = geo_dtype
torcwa.rcwa_geo.device = device
torcwa.rcwa_geo.Lx = L[0]
torcwa.rcwa_geo.Ly = L[1]
torcwa.rcwa_geo.nx = 300
torcwa.rcwa_geo.ny = 300
torcwa.rcwa_geo.grid()
torcwa.rcwa_geo.edge_sharpness = 500.

x_axis = torcwa.rcwa_geo.x.cpu()
y_axis = torcwa.rcwa_geo.y.cpu()

# layers
# layer0_geometry = torcwa.rcwa_geo.rectangle(Wx=180.,Wy=100.,Cx=L[0]/2.,Cy=L[1]/2.)
# layer0_eps = layer0_geometry*silicon_eps + (1.-layer0_geometry)
layer0_thickness = 250.

# %%
# Objective function

def objective_function(W):
    order_N = 10
    order = [order_N,order_N]

    sim = torcwa.rcwa(freq=1/lamb0,order=order,L=L,dtype=sim_dtype,device=device)
    sim.add_input_layer(eps=substrate_eps)
    sim.set_incident_angle(inc_ang=inc_ang,azi_ang=azi_ang)
    layer0_geometry = torcwa.rcwa_geo.rectangle(Wx=W[0],Wy=W[1],Cx=L[0]/2.,Cy=L[1]/2.)
    layer0_eps = layer0_geometry*silicon_eps + (1.-layer0_geometry)
    sim.add_layer(thickness=layer0_thickness,eps=layer0_eps)
    sim.solve_global_smatrix()
    txx = sim.S_parameters(orders=[0,0],direction='forward',port='transmission',polarization='xx',ref_order=[0,0])
    tyy = sim.S_parameters(orders=[0,0],direction='forward',port='transmission',polarization='yy',ref_order=[0,0])

    delta = torch.abs(tyy-txx)
    return delta

# %%
# Perform optimization
# optimizer parameters for ADAM optimizer
gar_initial = 1
gar = gar_initial
beta1 = 0.9
beta2 = 0.999
epsilon = 1.e-8
iter_max = 400

W = torch.tensor([100., 50.],dtype=geo_dtype,device=device)
momentum = torch.zeros_like(W)
velocity = torch.zeros_like(W)

W1_history = []
W2_history = []
delta_history = []

start_time = time.time()
for it in range(0,iter_max):
    W.requires_grad_(True)

    delta = objective_function(W)
    delta.backward()

    with torch.no_grad():
        W_gradient = W.grad
        W.grad = None

        W1_history.append(float(W[0].detach().cpu().numpy()))
        W2_history.append(float(W[1].detach().cpu().numpy()))
        delta_history.append(float(delta.detach().cpu().numpy()))

        momentum = (beta1*momentum + (1-beta1)*W_gradient)
        velocity = (beta2*velocity + (1-beta2)*(W_gradient**2))
        W += gar*(momentum / (1-beta1**(it+1))) / torch.sqrt((velocity / (1-beta2**(it+1))) + epsilon)
        W[W<50.] = 50.
        W[W>250.] = 250.
        gar -= gar_initial/iter_max

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Iteration:',it,'/ Delta:',int(1000*delta.detach().cpu().numpy())/1000,'/ Elapsed time:',str(int(elapsed_time))+' s')

# Export data
filename = 'Example5_data.mat'
ex5_data = {'W1_history':W1_history,'W2_history':W2_history,'delta_history':delta_history}
scipy.io.savemat(filename,ex5_data)
