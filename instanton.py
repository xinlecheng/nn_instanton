import scipy.special
import torch
from torch import pi
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import visualization
import numpy as np
from numpy import sqrt
import scipy

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size:tuple, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0]) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size[1], output_size)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

class NeuralNet1Hidden(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

#region
# class ActionPiece(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input:torch.Tensor, delta_t):
#         ctx.save_for_backward(input)
#         ctx.delta_t = delta_t
#         return (input[1]-input[0])**2/2/delta_t + (input[1]+input[0])**2/8*delta_t
#     @staticmethod
#     def backward(ctx, grad_output):
#         input, = ctx.saved_tensors()
#         delta_t = ctx.delta_t
#         grad_input = grad_output*torch.tensor(
#             [(input[1]+input[0])*delta_t/4 - (input[1]-input[0])/delta_t,
#              (input[1]+input[0])*delta_t/4 + (input[1]-input[0])/delta_t])
#         return grad_input

# class BoundaryCondition(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input:torch.Tensor, Lambda, bc:tuple):
#         ctx.save_for_backward(input)
#         ctx.Lambda = Lambda
#         ctx.bc = bc
#         return (input[0]-bc[0])**2/2*Lambda + (input[1]-bc[1])**2/2*Lambda
#     @staticmethod
#     def backward(ctx, grad_output):
#         input, = ctx.saved_tensors()
#         Lambda = ctx.Lambda
#         bc = ctx.bc
#         grad_input = grad_output*torch.tensor(
#             [(input[0]-bc[0])*Lambda, (input[1]-bc[1])*Lambda]
#         )
#endregion

class DummyFunc(nn.Module):
    def __init__(self, start, end):
        super().__init__()
        self.start = start
        self.end = end
    def forward(self, t):
        return self.start + t*(self.end-self.start) - (self.end - self.start)*torch.sin(2*pi*t)/(2*pi)

def rot_z(c, format='torch'):
    if format == 'numpy':
        return np.array([[np.cos(c), -np.sin(c)],[np.sin(c), np.cos(c)]]).astype(np.float64)
    elif format == 'torch':
        return torch.tensor([[np.cos(c), -np.sin(c)],[np.sin(c), np.cos(c)]]).to(torch.float32)

def functional_if(crit, true_return, false_return=None):
    if crit:
        return true_return
    else:
        return false_return

def kinetic_energy(v: torch.Tensor):
    '''
    v are the veocities of all particles, a rank 1 tensor
    '''
    mass_mat = 0.01*torch.eye(v.shape[0])
    return torch.matmul(v, torch.matmul(mass_mat, v))/2

def vint(x:torch.Tensor, y:torch.Tensor, format='torch'):
    a = 1.0
    b = 0.01
    if format == 'torch':
        return torch.exp(-a*torch.norm(x-y))/(torch.norm(x-y) + b)
    elif format == 'numpy':
        return np.exp(-a*np.linalg.norm(x-y))/(np.linalg.norm(x-y) + b)

class PotEnergy1P(nn.Module):
    def __init__(self, cutoff:float):
        super().__init__()
        mobile_pos = [np.array([1/2, sqrt(3)/6]), np.array([1, -sqrt(3)/3]),
                            np.array([3/2, sqrt(3)/6]), np.array([1, 2*sqrt(3)/3])]
        self.mobile_pos = mobile_pos + [np.matmul(rot_z(2*pi/3, format='numpy'), pos) for pos in mobile_pos] +\
                            [np.matmul(rot_z(4*pi/3, format='numpy'), pos) for pos in mobile_pos]
        self.num_mobile = len(self.mobile_pos)
        def ntopos(n1, n2):
            return np.array([1/2, sqrt(3)/6]) + n1*np.array([1,0]) + n2*np.array([-1/2,sqrt(3)/2])
        def close_to_any(vec, vec_list):
            for vec1 in vec_list:
                if np.all(np.isclose(vec, vec1)):
                    return True
            return False
        self.neighbors = [[torch.tensor(ntopos(n1,n2)).to(torch.float32)
                           for n1 in range(-4*int(cutoff), 4*int(cutoff)) for n2 in range(-4*int(cutoff), 4*int(cutoff)) 
                           if not close_to_any(ntopos(n1,n2), self.mobile_pos) and np.linalg.norm(ntopos(n1,n2)-self.mobile_pos[i]) < cutoff]
                           for i in range(self.num_mobile)]
        self.offset = torch.tensor(sum(functional_if(close_to_any(ntopos(n1,n2), self.mobile_pos), 1/2, 1)*\
                                       vint(self.mobile_pos[i], ntopos(n1,n2), format='numpy')
                                       for i in range(self.num_mobile)
                                       for n1 in range(-4*int(cutoff), 4*int(cutoff)) for n2 in range(-4*int(cutoff), 4*int(cutoff))
                                       if 0.01 < np.linalg.norm(ntopos(n1,n2)-self.mobile_pos[i]) < cutoff)).to(torch.float32)
        self.mobile_pos = [torch.tensor(pos).to(torch.float32) for pos in self.mobile_pos]     #convert self.mobile_pos to a list of torch.Tensor
    def forward(self, x: torch.Tensor):
        x_reshaped = x.reshape(x.shape[0]//2, 2)
        x_patched = torch.stack([x_i for x_i in x_reshaped]
                                + [torch.matmul(rot_z(2*pi/3), x_i) for x_i in x_reshaped]
                                + [torch.matmul(rot_z(4*pi/3), x_i) for x_i in x_reshaped])  # enlarge x according to symmetry
        return torch.stack([vint(x_patched[i], neighbor) 
                            for i in range(self.num_mobile) for neighbor in self.neighbors[i]]).sum() - self.offset

def pot_energy_2p(x: torch.Tensor):
    '''
    x are the positions of all particles, a rank 1 tensor
    '''
    x_reshaped = x.reshape(x.shape[0]//2, 2)
    x_patched = torch.stack([x_i for x_i in x_reshaped]
                            + [torch.matmul(rot_z(2*pi/3), x_i) for x_i in x_reshaped]
                            + [torch.matmul(rot_z(4*pi/3), x_i) for x_i in x_reshaped])  # enlarge x according to symmetry
    num_mobile = x_patched.shape[0]
    return torch.stack([vint(x_patched[i], x_patched[j]) 
                        for i in range(num_mobile) for j in range(num_mobile) if i!=j]).sum()/2
 
class ActionPiece(nn.Module):
    def __init__(self, delta_t):
        super().__init__()
        self.delta_t = delta_t
    def forward(self, x:torch.Tensor, pot_energy_1p:PotEnergy1P):
        #return (x[1]-x[0])**2/2/self.delta_t + (x[1]+x[0])**2/8*self.delta_t
        #return (0.01*(x[1]-x[0])**2/2/self.delta_t).sum() - (-1+torch.cos((x[1]+x[0])/2))*self.delta_t
        return self.delta_t*( kinetic_energy((x[1]-x[0])/self.delta_t) +
                              pot_energy_1p((x[1]+x[0])/2) + pot_energy_2p((x[1]+x[0])/2) )
    
class BoundaryCondition(nn.Module):
    def __init__(self, bc:torch.Tensor, Lambda:float = 10):
        super().__init__()
        self.bc = bc
        self.Lambda = Lambda
    def forward(self, x:torch.Tensor):
        return self.Lambda*((x-self.bc)**2).sum()/2 #self.Lambda*(x[0]-self.bc[0])**2/2 + self.Lambda*(x[1]-self.bc[1])**2/2

def action(x:torch.Tensor, pot_energy_1p:PotEnergy1P):
    num_t = x.shape[0]
    action_piece = ActionPiece(1/(num_t-1))
    boundary_condition = BoundaryCondition(
        torch.tensor([[1/2, sqrt(3)/6, 1, -sqrt(3)/3, 3/2, sqrt(3)/6, 1, 2*sqrt(3)/3],
                      [-1/2, sqrt(3)/6, 1, -sqrt(3)/3, 3/2, sqrt(3)/6, 1, 2*sqrt(3)/3]]), Lambda=50)
    bulk_action = torch.stack([action_piece(x[i:i+2], pot_energy_1p) for i in range(num_t-1)]).sum()
    boundary_action = boundary_condition(torch.stack([x[0], x[-1]]))
    return bulk_action + boundary_action


if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Hyper-parameters 
    input_size = 1
    hidden_size = (50,50)
    output_size = 8
    learning_rate = 0.001

    num_t = 51
    t_arr = [i/(num_t-1) for i in range(num_t)]
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    #model = NeuralNet1Hidden(input_size, 100, output_size).to(device)
    # dummy_model = DummyFunc(0, 2*pi)
    # traj_dummy = torch.stack([dummy_model(torch.tensor([t])) for t in t_arr])

    pot_energy_1p = PotEnergy1P(5.1)
    start = torch.tensor([1/2, sqrt(3)/6, 1, -sqrt(3)/3, 3/2, sqrt(3)/6, 1, 2*sqrt(3)/3]).to(torch.float32)
    end = torch.tensor([-1/2, sqrt(3)/6, 1, -sqrt(3)/3, 3/2, sqrt(3)/6, 1, 2*sqrt(3)/3]).to(torch.float32)
    print(pot_energy_1p(start) + pot_energy_2p(start))
    print(pot_energy_1p(end) + pot_energy_2p(end))
    # for i in range(12):
    #     print(min([torch.norm(neighbor).item() for neighbor in pot_energy_1p.neighbors[i]])) #2.081666
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    total_step = 1000
    error_curve = []

    for step in range(total_step):
        # Forward pass
        traj = torch.stack([model(torch.tensor([t])) for t in t_arr])
        loss = action(traj, pot_energy_1p)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (step+1) % 100 == 0:
            print ('Step [{}/{}], Loss: {:.4f}' 
                .format(step+1, total_step, loss.item()))
        error_curve.append((step, np.log10(loss.item())))
    
    torch.save(model.state_dict(), 'instanton_model_weights.pth')

    visualization.list_plot(error_curve, aspect_ratio=total_step/error_curve[0][1]/2)
    print(f"final loss: {10**(error_curve[-1][1])}")
    #print(f"dummy loss: {action(traj_dummy)}")
    num_t_plt = 201
    t_arr_plt = [i/(num_t_plt-1) for i in range(num_t_plt)]
    visualization.list_plot([(t,model(torch.tensor([t]))[0].item()) for t in t_arr_plt],
                             aspect_ratio = 1/(6*pi))
