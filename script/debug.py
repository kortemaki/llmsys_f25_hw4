import minitorch
import numpy as np
import pdb

backend = minitorch.TensorBackend(minitorch.CudaKernelOps)

def debug_layernorm_fw():
    inp = minitorch.tensor([[1,2,3,4,5,6,7,8]], backend=backend, requires_grad=True)
    gamma = minitorch.tensor([10,20,30,40,50,60,70,80], backend=backend, requires_grad=True)
    betta = minitorch.tensor([100,200,300,400,500,600,700,800], backend=backend, requires_grad=True)
    out = inp.layernorm(gamma, betta)
    print("layernorm output (yours, expected):")
    print(out)

    inp = minitorch.tensor([[1,2,3,4,5,6,7,8]], backend=backend, requires_grad=True)
    x = inp.contiguous()
    batch, dim = x.shape
    mean = x.mean(dim=1).view(batch, 1)
    variance = x.var(dim=1).view(batch, 1)
    x = (x - mean) / ((variance + 1e-8) ** 0.5)
    x = gamma * x + betta
    print(x)

def debug_layernorm_bw():
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

def debug_softmax_bw():
    inp = minitorch.tensor([[[[1],[2],[3],[4]]]], backend=backend, requires_grad=True)
    mask = minitorch.tensor(np.zeros((1,1,1,1)).tolist(), backend=backend, requires_grad=True)
    out_grad = minitorch.tensor([[[[.1],[.2],[.3],[.4]]]], backend=backend, requires_grad=True)
    soft_inp = inp.attn_softmax(mask)
    soft_inp.backward(out_grad)
    print("input gradients (yours, expected):")
    print(inp.grad)

    tsum = out_grad * soft_inp
    tsum = tsum.sum(dim=3).view(tsum.shape[0], tsum.shape[1], tsum.shape[2], 1)
    res = soft_inp * (out_grad - tsum)
    print(res)

if __name__ == '__main__':
    debug_layernorm_fw()
