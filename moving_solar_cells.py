from moving_cells_utils import environment
import time
import os
import datetime
import csv
import atexit

def main():
    """
    Call environment to set up UGV and solve opvp, the simulate ugv for TMAX steps and record the results.
    """
    log_dir = "moving_cells_utils/logs"
    os.makedirs(log_dir, exist_ok=True)
    csv_str = ".csv"
    date_time = datetime.datetime.now()
    stats_log = ("solver_routes_" + date_time.strftime("%d") + "_" + date_time.strftime("%m") + csv_str)
    stats_logfile = os.path.join(log_dir, stats_log)
    # open once, append mode; newline='' avoids blank lines on Windows
    stats_file = open(stats_logfile, mode='a', newline='', encoding='utf-8')
    stats_writer = csv.writer(stats_file, delimiter='|')
    # write header only if file is empty
    if os.path.getsize(stats_logfile) == 0:
        stats_writer.writerow(["e_harvest", "e_move", "e_overhead"])
        stats_file.flush()

    max_time = 720  # min
    start_time = time.perf_counter()
    env = environment.SingleUGVEnv(max_steps=max_time)
    print(time.perf_counter() - start_time)

    start_time = time.perf_counter()
    for t in range(max_time):
        e_harvest, e_move, e_overhead = env.step(t)
        stats_writer.writerow([e_harvest, e_move, e_overhead])
        stats_file.flush()
    print(time.perf_counter() - start_time)

    try:
        stats_file.close()
    except Exception:
        pass
    atexit.register(lambda: stats_file and not stats_file.closed and stats_file.close())

if __name__ == "__main__":
    main()