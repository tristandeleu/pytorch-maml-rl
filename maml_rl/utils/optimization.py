import torch
from torch.autograd import Variable

def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    p = Variable(b.data)
    r = Variable(b.data)
    x = torch.zeros_like(b).float()
    rdotr = torch.dot(r, r)

    for i in range(cg_iters):
        z = Variable(f_Ax(p).data)
        v = rdotr / torch.dot(p, z)
        x += v * p
        r -= v * z
        newrdotr = torch.dot(r, r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr.data[0] < residual_tol:
            break

    return Variable(x.data)
