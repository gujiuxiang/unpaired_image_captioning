from activations import *

def numeric_gradient(function, grad_output, *inputs):
    '''This uses the method of finite differences to
    compute the numeric gradient. Works for any function
    with multiple inputs and one output.'''
    epsilon = 1e-6
    grad_inputs = [[] for input in inputs]
    for i in xrange(len(inputs)):
        J = np.zeros((inputs[i].cpu().numpy().shape[0],
                      grad_output.cpu().numpy().shape[1],
                      inputs[i].cpu().numpy().shape[1]))
        for j in xrange(n):
            x1 = inputs[i].cpu().numpy()
            x2 = inputs[i].cpu().numpy()
            x1[:, j] -= epsilon
            x2[:, j] += epsilon
            inputs1 = list(inputs)
            inputs1[i] = torch.from_numpy(x1).cuda()
            inputs2 = list(inputs)
            inputs2[i] = torch.from_numpy(x2).cuda()
            output1 = function(*inputs1)
            output2 = function(*inputs2)
            J[:, :, j] = (output2 - output1).cpu().numpy() / (2*epsilon)
        grad_input = np.zeros((inputs[i].cpu().numpy().shape[0],
                               inputs[i].cpu().numpy().shape[1]))
        for b in xrange(inputs[i].cpu().numpy().shape[0]):
            grad_input[b, :] = J[b, :, :].transpose().dot(
                grad_output[b, :].cpu().numpy())
        grad_inputs[i] = torch.from_numpy(grad_input).cuda()
    return tuple(grad_inputs)

if __name__ == "__main__":
    # There is a gradient check in torch.autograd, but it doesn't seem to
    # be very informative and sometimes it segfaults (not sure why).
    # So I commented this out.
    #from torch.autograd import gradcheck
    #from torch.autograd import Variable
    #
    ## gradchek takes a tuple of tensor as input, check if your gradient
    ## evaluated with these tensors are close enough to numerical
    ## approximations and returns True if they all verify this condition.
    #input = (Variable(torch.randn(20,20).double().cuda(), requires_grad=True),)
    #test = gradcheck(Sparsemax(), input, eps=1e-6, atol=1e-4)
    #print(test)

    from torch.autograd import Variable
    n = 6
    batch_size = 3
    sf = SparsemaxFunction()

    z = Variable(torch.randn(batch_size, n).double().cuda(), requires_grad=True)
    p = Sparsemax()(z)
    dp = torch.randn(batch_size, n).double().cuda()
    p.backward(dp)
    dz = z.grad.data
    dz = dz.cpu().numpy()
    print dp.cpu().numpy()
    print dz

    dz_, = numeric_gradient(sf.forward,
                            dp,
                            z.data.cuda())
    dz_ = dz_.cpu().numpy()
    print dz_
    print np.linalg.norm(dz - dz_)


    print
    print


    #sf = ConstrainedSoftmaxFunction()
    sf = ConstrainedSparsemaxFunction()
    z = Variable(torch.randn(batch_size, n).double().cuda(), requires_grad=True)
    # This makes sure that each row of u has a sum larger than 1 (otherwise
    # the problems are infeasible).
    u = Variable(1./n + 1./n * torch.rand(batch_size, n).double().cuda(),
                 requires_grad=True)
    #z = Variable(torch.from_numpy(np.array([[1.,1.,1.,1.,1.,1.,1.,1.]])).cuda(),
    #             requires_grad=True)
    #u = Variable(torch.from_numpy(np.array([[.15,.167,1.,0.,1.,1.,0.,1.]])).cuda(),
    #             requires_grad=True)
    #p = ConstrainedSoftmax()(z, u)
    import pdb; pdb.set_trace()

    #############
    # z_sample = np.load("z_sample")[None, :]
    # u_sample = np.load("u_sample")[None, :]
    # batch_size = z_sample.shape[0]
    # n = z_sample.shape[1]
    # z = Variable(torch.from_numpy(z_sample).double().cuda(), requires_grad=True)
    # u = Variable(torch.from_numpy(u_sample).double().cuda(), requires_grad=True)
    #############

    p = ConstrainedSparsemax()(z, u)
    print("p:", p)
    print("u:", u)
    print("u >= p", torch.le(p, u))
    all_ones = torch.ones(batch_size, n).double().cuda()
    assert torch.equal(all_ones, torch.le(p, u).data.double().cuda()),"Probabilities are greater than the upper bounds!"
    assert (1 - (p <= u)).cpu().data.numpy().sum() == 0, pdb.set_trace()
    dp = torch.randn(batch_size, n).double().cuda()
    p.backward(dp)
    dz = z.grad.data
    dz = dz.cpu().numpy()
    du = u.grad.data
    du = du.cpu().numpy()
    print p.data.cpu().numpy()
    print p.data.cpu().numpy().sum(1)
    print dp.cpu().numpy()
    print dz
    print du

    dz_, du_ = numeric_gradient(sf.forward,
                                dp,
                                z.data.cuda(),
                                u.data.cuda())
    dz_ = dz_.cpu().numpy()
    du_ = du_.cpu().numpy()
    print dz_
    print du_
    print np.linalg.norm(dz - dz_)
    print np.linalg.norm(du - du_)

