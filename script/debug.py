import minitorch
import numpy as np
import pdb

if __name__ == "__main__":
    backend = minitorch.TensorBackend(minitorch.CudaKernelOps)

    f_out_grad = minitorch.tensor([[1,0,0,0]], backend=backend)
    
    inp = minitorch.tensor([[1,2,3,4]], backend=backend, requires_grad=True)
    gamma = minitorch.tensor([10,20,30,40], backend=backend, requires_grad=True)
    betta = minitorch.tensor([100,200,300,400], backend=backend, requires_grad=True)
    out = inp.layernorm(gamma, betta)
    out.backward(f_out_grad)
    print("input gradients (yours, expected):")
    print(inp.grad)
    
    f_means = inp.mean(dim=1)
    f_vars = inp.var(dim=1)
    f_stds = minitorch.tensor(np.sqrt(f_vars.to_numpy()).reshape(-1, 1).tolist(), backend=backend, requires_grad=True)

    xhat = (inp - f_means) / f_stds
    dxhat = f_out_grad * gamma
    f_betta_grad = f_out_grad.sum(dim=0)
    f_gamma_grad = (f_out_grad * xhat).sum(dim=0)
    dinp = dxhat.sum(dim=1) + xhat * (dxhat * xhat).sum(dim=1)
    dinp = dxhat - dinp / 4
    dinp = dinp / f_stds    
    print(dinp)

    print("gamma gradients (yours, expected):")
    print(gamma.grad)
    print(f_gamma_grad)

    print("betta gradients (yours, expected):")
    print(betta.grad)
    print(f_betta_grad)

    pdb.set_trace()
