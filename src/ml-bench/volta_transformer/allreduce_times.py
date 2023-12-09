#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import sys

def run(rank, size):
    """ Distributed function to be implemented later. """
    for H in [12288, 8192]:
        results_csv = ""
        for b in [1,2,4,8,16,32,64,128,256,512,1024,2048]:
            inT = torch.ones(b*H,dtype=torch.half).cuda(rank)

            for i in range(10):
                dist.all_reduce(inT)
            torch.cuda.synchronize()
            dist.barrier()

            s = time.time()
            for i in range(100):
                dist.all_reduce(inT)
            torch.cuda.synchronize()
            e = time.time()
            
            if rank == 0:
                results_csv += f"{b} & {H} & {((e - s)/100.)*1000}\n"

        if rank == 0:
            with open(f"results/allreduce_times-{H}.csv", "w") as f:
                f.write(results_csv)

def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    size = min(8, torch.cuda.device_count())
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
