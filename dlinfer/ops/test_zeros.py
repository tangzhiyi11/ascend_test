import torch
import torch_npu
import dlinfer
import dlinfer.graph
import dlinfer.ops as ext_ops

dlinfer.graph.config.enable_graph_mode = True

def zeros_test(x):
    inter = torch.sigmoid(x)
    y = torch.zeros_like(x)
    return x + inter

x = torch.randn(10, 10, device="npu", dtype=torch.bfloat16)

eager_out = zeros_test(x)
print(eager_out)


compiled = torch.compile(zeros_test, backend='atbgraph', dynamic=False, fullgraph=True)
compiled_out = compiled(x)
print(compiled_out)