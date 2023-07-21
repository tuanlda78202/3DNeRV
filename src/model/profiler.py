from src.model.hnerv3d import HNeRVMae
import torch.autograd.profiler as profiler
import torch

data = torch.rand(4, 3, 720, 1080).cuda()

model = HNeRVMae(
    img_size=(720, 1280),
    frame_interval=4,
    embed_dim=8,
    decode_dim=314,
    embed_size=(9, 16),
    scales=[5, 4, 2, 2],
    lower_width=6,
    reduce=3,
    ckpt_path="../ckpt/vit_s_k710_dl_from_giant.pth",
).cuda()

with profiler.profile(with_stack=True, profile_memory=True) as prof:
    output = model(data)

print(
    prof.key_averages(group_by_stack_n=5).table(
        sort_by="self_cpu_time_total", row_limit=5
    )
)


# Non-default profiler schedule allows user to turn profiler on and off
# on different iterations of the training loop;
# trace_handler is called every time a new trace becomes available
def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    # prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")


with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    # In this example with wait=1, warmup=1, active=2, repeat=1,
    # profiler will skip the first step/iteration,
    # start warming up on the second, record
    # the third and the forth iterations,
    # after which the trace will become available
    # and on_trace_ready (when set) is called;
    # the cycle repeats starting with the next step
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
    on_trace_ready=trace_handler
    # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    # used when outputting for tensorboard
) as p:
    for iter in range(N):
        code_iteration_to_profile(iter)
        # send a signal to the profiler that the next iteration has started
        p.step()
