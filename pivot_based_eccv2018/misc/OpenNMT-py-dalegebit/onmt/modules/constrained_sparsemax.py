import torch
from torch.autograd import Variable
import numpy as np
import pdb
# from activations import Softmax, Sparsemax

def constrained_sparsemax(z, u):
    '''Solve the problem:

    min_p 0.5*\|p - z\|^2
    s.t. p <= u
         p in simplex
    '''
    l = np.zeros_like(u)
    return double_constrained_sparsemax(z, l, u)

def double_constrained_sparsemax(z, l, u):
    '''Solve the problem:

    min_p 0.5*\|p - z\|^2
    s.t. l <= p <= u
         p in simplex

    This maps to Pardalos' canonical problem by making the transformations
    below.
    '''
    #assert round(np.sum(u), 3) >= 1.0 , "Invalid : sum(u) < 1.0"
    #assert round(np.sum(u), 3) >= 1.0 , pdb.set_trace()
    assert (u>=0).all(), "Invalid: u[i]<0 for some i"

    dtype = z.dtype
    z = z.astype('float64')
    l = l.astype('float64')
    u = u.astype('float64')
    a = .5 * (l - z)
    b = .5 * (u - z)
    c = np.ones_like(z)
    d = .5 * (1 - z.sum())
    x, tau, regions = solve_quadratic_problem(a, b, c, d)
    tau = -2*tau
    p = z - tau
    ind = np.nonzero(regions == 0)[0]
    p[ind] = l[ind]
    ind = np.nonzero(regions == 2)[0]
    p[ind] = u[ind]
    #print 2*x + z
    #p = 2*x + z
    #tau = -2*tau
    p = p.astype(dtype)
    return p, regions, tau, .5*np.dot(p-z, p-z)

def solve_quadratic_problem(a, b, c, d):
    '''Solve the problem:

    min_x sum_i c_i x_i^2
    s.t. sum_i c_i x_i = d
         a_i <= x_i <= b_i, for all i.

    by using Pardalos' algorithm:

    Pardalos, Panos M., and Naina Kovoor.
    "An algorithm for a singly constrained class of quadratic programs subject
    to upper and lower bounds." Mathematical Programming 46.1 (1990): 321-328.
    '''
    K = np.shape(c)[0]

    # Check for tight constraints.
    ind = np.nonzero(a == b)[0]
    if len(ind):
        x = np.zeros(K)
        regions = np.zeros(K, dtype=int)
        x[ind] = a[ind]
        regions[ind] = 0 # By convention.
        dd = d - c[ind].dot(x[ind])
        ind = np.nonzero(a < b)[0]
        if len(ind):
            x[ind], tau, regions[ind] = \
                solve_quadratic_problem(a[ind], b[ind], c[ind], dd)
        else:
            tau = 0. # By convention.
        return x, tau, regions

    # Sort lower and upper bounds and keep the sorted indices.
    sorted_lower = np.argsort(a)
    sorted_upper = np.argsort(b)
    slackweight = 0.
    tightsum = np.dot(a, c)
    k, l, level = 0, 0, 0
    right = -np.inf
    found = False
    while k < K or l < K:
        # Compute the estimate for tau.
        if level:
            tau = (d - tightsum) / slackweight
        if k < K:
            index_a = sorted_lower[k]
            val_a = a[index_a]
        else:
            val_a = np.inf
        if l < K:
            index_b = sorted_upper[l]
            val_b = b[index_b]
        else:
            val_b = np.inf

        left = right
        if val_a < val_b:
            # Next value comes from the a-list.
            right = val_a
        else:
            # Next value comes from the b-list.
            left = right
            right = val_b

        assert not level or tau >= left, pdb.set_trace()
        if (not level and d == tightsum) or (level and left <= tau <= right):
            # Found the right split-point!
            found = True
            break

        if val_a < val_b:
            tightsum -= a[index_a] * c[index_a]
            slackweight += c[index_a]
            level += 1
            k += 1
        else:
            tightsum += b[index_b] * c[index_b]
            slackweight -= c[index_b]
            level -= 1
            l += 1

    x = np.zeros(K)
    if not found:
        left = right
        right = np.inf

    regions = -np.ones(K, dtype=int)
    for i in xrange(K):
        if a[i] >= right:
            x[i] = a[i]
            regions[i] = 0
        elif b[i] <= left:
            x[i] = b[i]
            regions[i] = 2
        else:
            assert found and level
            x[i] = tau
            regions[i] = 1

    return x, tau, regions

if __name__ == "__main__":
    n = 6
    lower = 1./n * np.random.rand(n) # Uniform in [0, 1/n].
    upper = 1./n + 1./n * np.random.rand(n) # Uniform in [1/n, 2/n].
    z = np.random.randn(n)
    print 'z:', z
    print 'Lower:', lower
    print 'Upper:', upper
    p, _, tau, value = double_constrained_sparsemax(z, lower, upper)
    print 'p:', p
    print 'tau:', tau
    print 'value:', value
    tol = 1e-12
    # print 'sparsemax p:', Sparsemax()(Variable(torch.Tensor(z).view(1,-1)))
    assert np.all(lower <= p) and np.all(p <= upper) \
        and 1-tol <= sum(p) <= 1+tol

