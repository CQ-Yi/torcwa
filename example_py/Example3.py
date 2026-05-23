# Converted from example/Example3.ipynb
# Cell markers (# %%) are kept for IDE/debugger navigation.

# %%
'''
TORCWA Example3
Parametric sweep on rectangular meta-atom

'''
# Import
import numpy as np
import torch
import scipy.io
import time

import torcwa
import Materials

# Hardware
# If GPU support TF32 tensor core, the matmul operation is faster than FP32 but with less precision.
# If you need accurate operation, you have to disable the flag below.
torch.backends.cuda.matmul.allow_tf32 = False
sim_dtype = torch.complex64
geo_dtype = torch.float32
device = torch.device('cpu')

# Simulation environment
# light
lamb0 = torch.tensor(532.,dtype=geo_dtype,device=device)    # nm
inc_ang = 0.*(np.pi/180)                    # radian
azi_ang = 0.*(np.pi/180)                    # radian

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
torcwa.rcwa_geo.edge_sharpness = 1000.

x_axis = torcwa.rcwa_geo.x.cpu()
y_axis = torcwa.rcwa_geo.y.cpu()

# layers
layer0_thickness = 300.

# %%
# Generate and perform simulation
order_N = 20
order = [order_N,order_N]
Wx = torch.linspace(50.,250.,11,dtype=geo_dtype,device=device)
Wy = torch.linspace(50.,250.,11,dtype=geo_dtype,device=device)

start_time = time.time()
txx = torch.zeros((11,11),dtype=sim_dtype,device=device)
for Wx_ind in range(len(Wx)):
    for Wy_ind in range(len(Wy)):
        sim = torcwa.rcwa(freq=1/lamb0,order=order,L=L,dtype=sim_dtype,device=device)
        sim.add_input_layer(eps=substrate_eps)
        sim.set_incident_angle(inc_ang=inc_ang,azi_ang=azi_ang)
        layer0_geometry = torcwa.rcwa_geo.rectangle(Wx=Wx[Wx_ind],Wy=Wy[Wy_ind],Cx=L[0]/2.,Cy=L[1]/2.)
        layer0_eps = layer0_geometry*silicon_eps + (1.-layer0_geometry)
        sim.add_layer(thickness=layer0_thickness,eps=layer0_eps)
        sim.solve_global_smatrix()
        txx[Wx_ind,Wy_ind] = sim.S_parameters(orders=[0,0],direction='forward',port='transmission',polarization='xx',ref_order=[0,0])
    print(str(int((Wx_ind+1)/len(Wx)*10000)/100)+' % Completed.')
end_time = time.time()
elapsed_time = end_time - start_time
print('Elapsed time: '+str(int(elapsed_time*100)/100)+' s')

# %%
# Export spectrum data
filename = 'Example3_spectrum_data_XeonGold5118_CPUonly_64bit_order_'+str(order_N)+'.mat'
# filename = 'Example3_spectrum_data_XeonGold5118_RTX3090_64bit_FP32_order_'+str(order_N)+'.mat'
# filename = 'Example3_spectrum_data_XeonGold5118_RTX3090_64bit_TF32_order_'+str(order_N)+'.mat'
# filename = 'Example3_spectrum_data_XeonGold5118_RTX3090_128bit_order_'+str(order_N)+'.mat'

ex3_data = {'Wx':Wx.cpu().numpy(),'Wy':Wy.cpu().numpy(),'txx':txx.cpu().numpy(),'elapsed_time':elapsed_time}
scipy.io.savemat(filename,ex3_data)
