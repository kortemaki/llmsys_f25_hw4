import minitorch

if __name__ == __main__():
    backend = minitorch.TensorBackend(minitorch.CudaKernelOps)

    inp = minitorch.tensor([[1,2,3,4]], backend=backend, requires_grad=True)
    gamma = minitorch.tensor([10,20,30,40], backend=backend, requires_grad=True)
    betta = minitorch.tensor([100,200,300,400], backend=backend, requires_grad=True)
    out = inp.layernorm(gamma, betta)
    out.backward(minitorch.tensor([[1,0,0,0]], backend=backend))
    print(inp.grad)
    print("""[
        [2.683282 -3.577709 -0.894427 1.788854]]""")
