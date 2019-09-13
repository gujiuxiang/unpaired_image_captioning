import torch
from torch.autograd import Function
from torch.nn import Module
from constrained_sparsemax import constrained_sparsemax
import numpy as np
import pdb
np.set_printoptions(threshold=np.nan)

def project_onto_simplex(a, radius=1.0):
    '''Project point a to the probability simplex.
    Returns the projected point x and the residual value.'''
    x0 = a.copy()
    d = len(x0);
    ind_sort = np.argsort(-x0)
    y0 = x0[ind_sort]
    ycum = np.cumsum(y0)
    val = 1.0/np.arange(1,d+1) * (ycum - radius)
    ind = np.nonzero(y0 > val)[0]
    rho = ind[-1]
    tau = val[rho]
    y = y0 - tau
    ind = np.nonzero(y < 0)
    y[ind] = 0
    x = x0.copy()
    x[ind_sort] = y
    return x, tau, .5*np.dot(x-a, x-a)

def constrained_softmax(z, u):
    assert round(np.sum(u), 5) >= 1.0, pdb.set_trace()
    assert (u>=0).all(), "Invalid: u[i]<0 for some i"
    p = np.zeros_like(z)
    active = np.ones_like(z)
    nz = np.nonzero(u)[0]
    z = z[nz]
    u = u[nz]
    active[nz] = 0.
    z -= np.max(z)
    e_z = np.exp(z)
    Z = e_z.sum()
    # if Z==0:
    #   return p, active, s
    ind = np.argsort(-e_z / u)
    s = 0.
    for i in ind:
        # Temporary fix for underflow in Z
        if round(Z, 12) == 0.0: Z = 0.000001
        val = e_z[i] * (1-s) / Z
        if val > u[i]:
            val = u[i]
            Z -= e_z[i]
            s += val
            active[nz[i]] = 1.
        p[nz[i]] = val
    #if np.any(np.isnan(p)):
    #    import pdb; pdb.set_trace()
    return p, active, s

class SoftmaxFunction(Function):
    def forward(self, input):
        e_z = input.exp()
        Z = e_z.sum(1)
        output = e_z / Z.expand_as(e_z)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        avg = (grad_output * output).sum(1)
        grad_input = output * (grad_output - avg.expand_as(grad_output))
        return grad_input

class Softmax(Module):
    def forward(self, input):
        return SoftmaxFunction()(input)

class SparsemaxFunction(Function):
    def forward(self, input):
        # TODO: Make an implementation directly with torch tensors,
        # not requiring numpy.
        # Example:
        # z_sorted, ind_sort = (-input).sort(dim=1, descending=True)
        # z_cum = z_sorted.cumsum(dim=1)
        # r = torch.arange(1, 1+z_sorted.size(1))
        # if input.is_cuda():
        #     r = r.cuda()
        # val = 1.0 / r.expand_as(z_cum) * (z_cum - 1.)
        # ...
        np_input = input.cpu().numpy()
        probs = np.zeros_like(np_input)
        for i in xrange(np_input.shape[0]):
            probs[i,:], tau, _ = project_onto_simplex(np_input[i,:])
        output = torch.from_numpy(probs)
        if input.is_cuda:
            output = output.cuda()
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        probs = output.cpu().numpy()
        supp = np.array(probs > 0., dtype=probs.dtype)
        np_grad_output = grad_output.cpu().numpy()
        avg = np.sum(np_grad_output * supp, 1) / np.sum(supp, 1)
        np_grad_input = supp * (np_grad_output - np.tile(avg[:,None],
                                                         [1, supp.shape[1]]))
        grad_input = torch.from_numpy(np_grad_input)
        if grad_output.is_cuda:
           grad_input  = grad_input.cuda()
        return grad_input

class Sparsemax(Module):
    def forward(self, input):
        return SparsemaxFunction()(input)

class ConstrainedSoftmaxFunction(Function):
    def forward(self, input1, input2):
        z = input1.cpu().numpy()
        u = input2.cpu().numpy()
        probs = np.zeros_like(z)
        active = np.zeros_like(z)
        s = np.zeros_like(z[:,0])
        for i in xrange(z.shape[0]):
            probs[i,:], active[i,:], s[i] = constrained_softmax(z[i], u[i])
        probs = torch.from_numpy(probs)
        active = torch.from_numpy(active)
        s = torch.from_numpy(s)
        if input1.is_cuda:
            probs = probs.cuda()
            active = active.cuda()
            s = s.cuda()
        self.save_for_backward(probs)
        self.saved_intermediate = active, s # Not sure this is safe.
        return probs

        #z = input1
        #u = input2
        #e_z = z.exp()
        #Z = e_z.sum(1)
        #probs = e_z / Z.expand_as(e_z)
        #active = (probs > u).type(probs.type())
        #s = (active * u).sum(1)
        #Z = ((1. - active) * e_z).sum(1) / (1-s)
        #probs = active * u + (1. - active) * (e_z / Z.expand_as(z))
        #output = probs
        #self.save_for_backward(output)
        #self.saved_intermediate = active, s # Not sure this is safe.
        #return output

    def backward(self, grad_output):
        output, = self.saved_tensors
        active, s = self.saved_intermediate
        probs = output
        m = ((1. - active) * probs * grad_output).sum(1) / (1. - s)
        m = m.squeeze(-1) # This is needed for back-compatibility with pytorch 0.1.x.
        # If all are active, then sum(u) = 1, s = 1, p = u, so we need to do
        # the following to avoid nans.
        ind = active.sum(1) == active.size(1)
        m[ind] = 0.
        grad_z = (1. - active) * probs * \
                 (grad_output - m.unsqueeze(1).expand_as(active))
        grad_u = active * (grad_output - m.unsqueeze(1).expand_as(active))
        grad_input1 = grad_z
        grad_input2 = grad_u
        #if np.any(np.isnan(grad_z.cpu().numpy())):
        #    import pdb; pdb.set_trace()
        #if np.any(np.isnan(grad_u.cpu().numpy())):
        #    import pdb; pdb.set_trace()
        return grad_input1, grad_input2

class ConstrainedSoftmax(Module):
    def forward(self, input1, input2):
        return ConstrainedSoftmaxFunction()(input1, input2)

class ConstrainedSparsemaxFunction(Function):
    def forward(self, input1, input2):
        z = input1.cpu().numpy()
        u = input2.cpu().numpy()
        #print("z:", z)
        #print("u:", u)
        probs = np.zeros_like(z)
        regions = np.zeros_like(z)
        for i in xrange(z.shape[0]):
            probs[i,:], regions[i,:], _, _ = constrained_sparsemax(z[i], u[i])
            assert np.all(probs[i, :] == probs[i, :]), pdb.set_trace()
        probs = torch.from_numpy(probs)
        regions = torch.from_numpy(regions)
        if input1.is_cuda:
            probs = probs.cuda()
            regions = regions.cuda()
        self.save_for_backward(probs)
        self.saved_intermediate = regions, # Not sure this is safe.
        return probs

    def backward(self, grad_output):
        output, = self.saved_tensors
        regions, = self.saved_intermediate
        probs = output
        regions = regions.cpu().numpy() # TODO: do everything with tensors.
        r0 = np.array(regions == 0, dtype=regions.dtype)
        r1 = np.array(regions == 1, dtype=regions.dtype)
        r2 = np.array(regions == 2, dtype=regions.dtype)
        np_grad_output = grad_output.cpu().numpy()
        #import pdb; pdb.set_trace()
        avg = np.sum(np_grad_output * r1, 1) / np.sum(r1, 1)
        np_grad_input1 = r1 * (np_grad_output - np.tile(avg[:,None],
                                                        [1, r1.shape[1]]))
        np_grad_input2 = r2 * (np_grad_output - np.tile(avg[:,None],
                                                        [1, r2.shape[1]]))
        #print("grad_output:", np_grad_output)
        #print("grad1:", np_grad_input1)
        #print("grad2:", np_grad_input2)
        ind = np.nonzero(np.sum(r1, 1) == 0)[0]
        for i in ind:
            np_grad_input1[i, :] = 0.
            np_grad_input2[i, :] = 0.
        #print("grad_output:", np_grad_output)
        #print("grad1:", np_grad_input1)
        #print("grad2:", np_grad_input2)
        #pdb.set_trace()
        assert np.all(np_grad_input1 == np_grad_input1), pdb.set_trace()
        assert np.all(np_grad_input2 == np_grad_input2), pdb.set_trace()
        grad_input1 = torch.from_numpy(np_grad_input1)
        grad_input2 = torch.from_numpy(np_grad_input2)
        if grad_output.is_cuda:
           grad_input1  = grad_input1.cuda()
           grad_input2  = grad_input2.cuda()
        return grad_input1, grad_input2


class ConstrainedSparsemax(Module):
    def forward(self, input1, input2):
        return ConstrainedSparsemaxFunction()(input1, input2)

