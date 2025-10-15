import minitorch
import numpy as np
import pdb

if __name__ == "__main__":
    backend = minitorch.TensorBackend(minitorch.CudaKernelOps)

    f_out_grad = minitorch.tensor([[.1,.2,.3,.4,.5,.6,.7,.8]], backend=backend)

    inp = minitorch.tensor([[1,2,3,4,5,6,7,8]], backend=backend, requires_grad=True)
    gamma = minitorch.tensor([10,20,30,40,50,60,70,80], backend=backend, requires_grad=True)
    betta = minitorch.tensor([100,200,300,400,500,600,700,800], backend=backend, requires_grad=True)
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
    dinp = dxhat - dinp / f_out_grad.shape[1]
    dinp = dinp / f_stds
    print(dinp)

    print("gamma gradients (yours, expected):")
    print(gamma.grad)
    print(f_gamma_grad)

    print("betta gradients (yours, expected):")
    print(betta.grad)
    print(f_betta_grad)

    (_, _, _, vars, means) = out.history.ctx.saved_values
    print("vars (yours, expected):")
    print(vars)
    print(f_vars)

    print("means (yours, expected):")
    print(means)
    print(f_means)

    pdb.set_trace()
