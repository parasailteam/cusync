import torch
import sys
import time

M = int(sys.argv[1])
N = int(sys.argv[2])
K = int(sys.argv[3])

a = torch.ones((M, K), dtype=torch.half).cuda()
b = torch.ones((K, N), dtype=torch.half).cuda()
#c = torch.ones((M, N), dtype=torch.half).cuda()
# d = torch.ones((N, L), dtype=torch.half).cuda()
#e = torch.ones([M, L], dtype=torch.half).cuda()

def matmul(a, b):
    return a@b

def run(func):
    for i in range(10):
        c = func(a,b)

    torch.cuda.synchronize()

    epochs = 20
    start = time.time_ns()

    for i in range(epochs):
        c = func(a,b)

    torch.cuda.synchronize()
    end = time.time_ns()

    print((end-start)/epochs/1e3)

# run(matmul)

a = torch.ones((2, M, K), dtype=torch.half).cuda()
b = torch.ones((2, K, N), dtype=torch.half).cuda()

# run(lambda x,y: torch.bmm(x,y))