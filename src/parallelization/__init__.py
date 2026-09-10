import os
import time
import subprocess
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m {int(s)}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m"

def parallel_runs(func, args_list, max_workers = None, info = "INFO", mpContext = None, show_job_progress=False):
    
    total = len(args_list)
    pending = list(enumerate(args_list))
    futures = {}
    results = [None] * total
    completed = 0
    start = time.perf_counter()

    # get the number of cores
    # do not take numbers higher than actual numbers of cores
    if max_workers is None:
        max_workers = os.cpu_count()
    else:
        max_workers = min(max_workers, os.cpu_count())
        

    # mp_context lets a caller force e.g. multiprocessing.get_context("spawn")
    # instead of the platform default (fork on Linux). This matters whenever
    # func relies on a library that initializes a multithreaded runtime at
    # import time (JAX/XLA being the relevant case here): forking after that
    # runtime is up is unsafe and can deadlock, whereas spawn re-imports the
    # module cleanly in each worker. Default None keeps existing callers
    # (e.g. the subprocess-based __main__ demo below) on the cheaper default.
    with ProcessPoolExecutor(max_workers=max_workers, mp_context = (get_context(mpContext) if mpContext else None)) as executor:
        # Launch the first batch
        while len(futures) < max_workers and pending:
            idx, args = pending.pop(0)
            futures[executor.submit(func, *args)] = idx

        # Print initial status (time = 0.0s)
        current_time = time.strftime("%Hh%Mm%Ss") 
        last_time_str = format_time(0.0)
        print(f"{info}Idle: {len(pending)},  Running: {len(futures)},  Completed: {completed} [ current time: {current_time} ]")

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)

            for future in done:
                idx = futures.pop(future)
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = e
                completed += 1
                if pending:
                    new_idx, new_args = pending.pop(0)
                    futures[executor.submit(func, *new_args)] = new_idx

            # Only print if the formatted time has changed (group by time slot)
            elapsed = time.perf_counter() - start
            time_str = format_time(elapsed)
            if time_str != last_time_str:
                print(f"{info}Idle: {len(pending)},  Running: {len(futures)},  Completed: {completed} [ {time_str} ]")
                last_time_str = time_str

    # Final print (ensures we show the very last state)
    final_time = format_time(time.perf_counter() - start)
    if final_time != last_time_str or completed == total:
        print(f"{info}Idle: {len(pending)},  Running: {len(futures)},  Completed: {completed} [ {final_time} ]")
    return results



if __name__ == "__main__":

    import sys

    def run_script(index):
        log_filename = f"logs/log{index}.log"
        with open(log_filename, "w") as log_file:
            cmd = [sys.executable, "test_func.py", str(index)]
            return subprocess.run(cmd, stdout=log_file).returncode # stdout -> log file, stderr -> terminal (inherit)

    args_list = [(i,) for i in range(100)]
    results = parallel_runs(run_script, args_list, max_workers=None, info="Testing jobs")