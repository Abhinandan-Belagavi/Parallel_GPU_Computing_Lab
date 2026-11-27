
import argparse
import time
import torch
from torch.profiler import profile, ProfilerActivity

def run_workload(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available. Check your PyTorch CUDA installation.")

    device = torch.device("cuda")
    torch.cuda.set_device(device)

    print("\nEXPERIMENT 7: PROFILING AND GPU UTILIZATION")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Mode:", args.mode)
    print("Rows per batch:", args.rows)
    print("Columns:", args.cols)
    print("Number of batches:", args.batches)
    print("Repetitions:", args.reps)

    # Warm-up CUDA before measurement
    a = torch.randn(args.rows, args.cols)
    b = torch.randn(args.cols, args.rows)

    for _ in range(2):
        x = a.to(device) @ b.to(device)
        torch.cuda.synchronize()

    stream = torch.cuda.Stream() if args.mode == "optimized" else None

    def execute_batch():
        checksum = 0.0

        for _ in range(args.batches):
            # Generate input on CPU
            host_a = torch.randn(args.rows, args.cols)
            host_b = torch.randn(args.cols, args.rows)

            if args.mode == "optimized":
                # Pinned memory supports asynchronous transfer
                host_a = host_a.pin_memory()
                host_b = host_b.pin_memory()

                with torch.cuda.stream(stream):
                    gpu_a = host_a.to(device, non_blocking=True)
                    gpu_b = host_b.to(device, non_blocking=True)

                    result = torch.mm(gpu_a, gpu_b)
                    result = torch.mul(result, 0.5)
                    result = torch.add(result, 1.0)
                    result = torch.clamp_min(result, 0)
                    batch_sum = torch.sum(result)

            else:
                # Baseline: ordinary host memory transfers
                gpu_a = host_a.to(device)
                gpu_b = host_b.to(device)

                result = torch.mm(gpu_a, gpu_b)
                result = torch.mul(result, 0.5)
                result = torch.add(result, 1.0)
                result = torch.clamp_min(result, 0)
                batch_sum = torch.sum(result)

            checksum += batch_sum.item()

        if stream is not None:
            stream.synchronize()
        else:
            torch.cuda.synchronize()

        return checksum

    # Actual workload timing
    total_start = time.perf_counter()

    for _ in range(args.reps):
        checksum = execute_batch()

    torch.cuda.synchronize()
    total_end = time.perf_counter()

    total_ms = (total_end - total_start) * 1000
    total_batches = args.batches * args.reps
    per_batch = total_ms / total_batches
    throughput = total_batches / (total_ms / 1000)

    print("\n========== RESULTS ==========")
    print(f"Total time:       {total_ms:.3f} ms")
    print(f"Time per batch:   {per_batch:.3f} ms")
    print(f"Throughput:       {throughput:.2f} batches/second")
    print(f"Checksum:         {checksum:.6f}")
    print("=============================")

    if args.profile:
        print("\nStarting PyTorch Profiler...")

        activities = [
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA
        ]

        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True
        ) as prof:
            execute_batch()
            torch.cuda.synchronize()

        trace_name = f"trace_{args.mode}.json"
        prof.export_chrome_trace(trace_name)

        print(f"\nProfiler trace saved: {trace_name}")
        print("\nTop profiler operations:")

        print(
            prof.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=15
            )
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 7 - GPU Profiling and Utilization"
    )

    parser.add_argument(
        "--mode",
        choices=["baseline", "optimized"],
        default="baseline"
    )

    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--profile", action="store_true")

    args = parser.parse_args()
    run_workload(args)

