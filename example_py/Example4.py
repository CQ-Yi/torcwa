# Converted from example/Example4.ipynb
# Cell markers (# %%) are kept for IDE/debugger navigation.

# %%
'''
TORCWA Example4
Gradient calculation of cylindrical meta-atom

'''
# Import
import numpy as np
import torch
import scipy.io

import torcwa

# Hardware
# If GPU support TF32 tensor core, the matmul operation is faster than FP32 but with less precision.
# If you need accurate operation, you have to disable the flag below.
torch.backends.cuda.matmul.allow_tf32 = False
sim_dtype = torch.complex128
geo_dtype = torch.float64
device = torch.device('cuda')

# Simulation environment
# light
lamb0 = torch.tensor(473.,dtype=geo_dtype,device=device)    # nm
inc_ang = 0.*(np.pi/180)                # radian
azi_ang = 0.*(np.pi/180)                # radian

# material
substrate_eps = 1.46**2
SiN_eps = 2.0709**2

# geometry
'''
    For accurate calculation of gradient through shape deviation,
    nx and ny should be much finer or edge sharpness should be lower
'''
L = [300., 300.]            # nm / nm
torcwa.rcwa_geo.dtype = geo_dtype
torcwa.rcwa_geo.device = device
torcwa.rcwa_geo.Lx = L[0]
torcwa.rcwa_geo.Ly = L[1]
torcwa.rcwa_geo.nx = 1500
torcwa.rcwa_geo.ny = 1500
torcwa.rcwa_geo.grid()
torcwa.rcwa_geo.edge_sharpness = 500. 

x_axis = torcwa.rcwa_geo.x.cpu()
y_axis = torcwa.rcwa_geo.y.cpu()

# layers
layer0_thickness = 600.

# %%
# Numerical gradient 
# Generate and perform simulation
order_N = 15
order = [order_N,order_N]
sampling = 41
R = torch.linspace(85.,105.,sampling,dtype=geo_dtype,device=device)
dR = 0.005

txx = []
txx_p = []
txx_m = []
R_grad_ex = torch.zeros(sampling,dtype=geo_dtype,device=device)
for R_ind in range(len(R)):
    # center
    sim = torcwa.rcwa(freq=1/lamb0,order=order,L=L,dtype=sim_dtype,device=device)
    sim.add_input_layer(eps=substrate_eps)
    sim.set_incident_angle(inc_ang=inc_ang,azi_ang=azi_ang)
    layer0_geometry = torcwa.rcwa_geo.circle(R=R[R_ind],Cx=L[0]/2.,Cy=L[1]/2.)
    layer0_eps = layer0_geometry*SiN_eps + (1.-layer0_geometry)
    sim.add_layer(thickness=layer0_thickness,eps=layer0_eps)
    sim.solve_global_smatrix()
    txx.append(sim.S_parameters(orders=[0,0],direction='forward',port='transmission',polarization='xx',ref_order=[0,0]))

    # +dR
    sim = torcwa.rcwa(freq=1/lamb0,order=order,L=L,dtype=sim_dtype,device=device)
    sim.add_input_layer(eps=substrate_eps)
    sim.set_incident_angle(inc_ang=inc_ang,azi_ang=azi_ang)
    layer0_geometry = torcwa.rcwa_geo.circle(R=R[R_ind]+dR,Cx=L[0]/2.,Cy=L[1]/2.)
    layer0_eps = layer0_geometry*SiN_eps + (1.-layer0_geometry)
    sim.add_layer(thickness=layer0_thickness,eps=layer0_eps)
    sim.solve_global_smatrix()
    txx_p.append(sim.S_parameters(orders=[0,0],direction='forward',port='transmission',polarization='xx',ref_order=[0,0]))

    # -dR
    sim = torcwa.rcwa(freq=1/lamb0,order=order,L=L,dtype=sim_dtype,device=device)
    sim.add_input_layer(eps=substrate_eps)
    sim.set_incident_angle(inc_ang=inc_ang,azi_ang=azi_ang)
    layer0_geometry = torcwa.rcwa_geo.circle(R=R[R_ind]-dR,Cx=L[0]/2.,Cy=L[1]/2.)
    layer0_eps = layer0_geometry*SiN_eps + (1.-layer0_geometry)
    sim.add_layer(thickness=layer0_thickness,eps=layer0_eps)
    sim.solve_global_smatrix()
    txx_m.append(sim.S_parameters(orders=[0,0],direction='forward',port='transmission',polarization='xx',ref_order=[0,0]))

txx = torch.cat(txx)
txx_p = torch.cat(txx_p)
txx_m = torch.cat(txx_m)

R_grad = (torch.abs(txx_p)**2 - torch.abs(txx_m)**2) / (2*dR)

filename = 'Example4_numerical_gradient_data.mat'
ex4_data = {'R':R.cpu().numpy(),'txx':txx.cpu().numpy(),'txx_p':txx_p.cpu().numpy(),'txx_m':txx_m.cpu().numpy(),'R_grad':R_grad.cpu().numpy()}
scipy.io.savemat(filename,ex4_data)

# %%
# Exact gradient
# Generate and perform simulation
R_grad = torch.zeros(sampling,dtype=geo_dtype,device=device)
for R_ind in range(len(R)):
    R_now = R[R_ind]
    R_now.requires_grad_(True)
    sim = torcwa.rcwa(freq=1/lamb0,order=order,L=L,dtype=sim_dtype,device=device,stable_eig_grad=False)
    sim.add_input_layer(eps=substrate_eps)
    sim.set_incident_angle(inc_ang=inc_ang,azi_ang=azi_ang)
    layer0_geometry = torcwa.rcwa_geo.circle(R=R_now,Cx=L[0]/2.,Cy=L[1]/2.)
    layer0_eps = layer0_geometry*SiN_eps + (1.-layer0_geometry)
    sim.add_layer(thickness=layer0_thickness,eps=layer0_eps)
    sim.solve_global_smatrix()
    Txx = torch.abs(sim.S_parameters(orders=[0,0],direction='forward',port='transmission',polarization='xx',ref_order=[0,0]))**2
    Txx.backward()

    with torch.no_grad():
        R_grad[R_ind] = R_now.grad
        R_now.grad = None

filename = 'Example4_exact_gradient_data.mat'
ex4_data = {'R':R.cpu().numpy(),'R_grad':R_grad.cpu().numpy()}
scipy.io.savemat(filename,ex4_data)

# %%
# Stabilized gradient
param_list = [10.**(-10), None]

for pl in param_list:
    torcwa.Eig.broadening_parameter = pl

    # Generate and perform simulation
    R_grad = torch.zeros(sampling,dtype=geo_dtype,device=device)
    for R_ind in range(len(R)):
        R_now = R[R_ind]
        R_now.requires_grad_(True)
        sim = torcwa.rcwa(freq=1/lamb0,order=order,L=L,dtype=sim_dtype,device=device,stable_eig_grad=True)
        sim.add_input_layer(eps=substrate_eps)
        sim.set_incident_angle(inc_ang=inc_ang,azi_ang=azi_ang)
        layer0_geometry = torcwa.rcwa_geo.circle(R=R_now,Cx=L[0]/2.,Cy=L[1]/2.)
        layer0_eps = layer0_geometry*SiN_eps + (1.-layer0_geometry)
        sim.add_layer(thickness=layer0_thickness,eps=layer0_eps)
        sim.solve_global_smatrix()
        Txx = torch.abs(sim.S_parameters(orders=[0,0],direction='forward',port='transmission',polarization='xx',ref_order=[0,0]))**2
        Txx.backward()

        with torch.no_grad():
            R_grad[R_ind] = R_now.grad
            R_now.grad = None

    filename = 'Example4_stabilized_gradient_data_param_'+str(pl)+'.mat'
    ex4_data = {'R':R.cpu().numpy(),'R_grad':R_grad.cpu().numpy()}
    scipy.io.savemat(filename,ex4_data)
