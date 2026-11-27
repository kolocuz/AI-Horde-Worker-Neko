# pop-stats.py
# Calculate node pop stats from the local worker log file.
# Usage: pop-stats.py [-h] [--today] [--yesterday]
import argparse
import datetime
import glob
import mmap
import re

from tqdm import tqdm

# Location of stable horde worker bridge log
LOG_FILE = "logs/bridge*.log"

# TIME PERIODS
PERIOD_ALL = 0
PERIOD_TODAY = 1
PERIOD_YESTERDAY = 2
PERIOD_HOUR = 3

# regex to identify model lines
REGEX = re.compile(r".*(\d\d\d\d-\d\d-\d\d \d\d:\d\d).* Job pop took (\d+\.\d+).*node: (.*)\)")


class LogStats:
    def __init__(self, period=PERIOD_ALL, logfile=LOG_FILE):
        self.logfile = logfile
        self.period = period
        self.data = {}

    def get_date(self):
        # Dates in log format for filtering
        if self.period == PERIOD_TODAY:
            adate = datetime.datetime.now()
            adate = adate.strftime("%Y-%m-%d")
        elif self.period == PERIOD_YESTERDAY:
            adate = datetime.datetime.now() - datetime.timedelta(1)
            adate = adate.strftime("%Y-%m-%d")
        elif self.period == PERIOD_HOUR:
            # Use rolling 1-hour cutoff; matching handled in parse_log.
            adate = datetime.datetime.now() - datetime.timedelta(hours=1)
        else:
            adate = None
        return adate

    def get_num_lines(self, file_path):
        with open(file_path, "rb") as fp:
            size = fp.seek(0, 2)
            if size == 0:
                return 0
            fp.seek(0)
            with mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ) as buf:
                lines = 0
                while buf.readline():
                    lines += 1
                return lines

    def parse_log(self):
        # Identify all log files and total number of log lines
        total_log_lines = sum(self.get_num_lines(logfile) for logfile in glob.glob(self.logfile))
        progress = tqdm(total=total_log_lines, leave=True, unit=" lines", unit_scale=True)
        hour_cutoff = datetime.datetime.now() - datetime.timedelta(hours=1) if self.period == PERIOD_HOUR else None
        period_date = self.get_date() if self.period in {PERIOD_TODAY, PERIOD_YESTERDAY} else None
        for logfile in glob.glob(self.logfile):
            with open(logfile, "rt", encoding="utf-8", errors="ignore") as infile:
                for line in infile:
                    # Grab the lines we're interested in
                    if regex := REGEX.match(line):
                        if self.period == PERIOD_HOUR:
                            try:
                                line_time = datetime.datetime.strptime(regex.group(1), "%Y-%m-%d %H:%M")
                            except ValueError:
                                continue
                            if hour_cutoff and line_time < hour_cutoff:
                                continue
                        elif self.period and period_date:
                            line_date = regex.group(1).split()[0]
                            if period_date != line_date:
                                continue

                        # Extract pop time and node
                        pop_time = regex.group(2)
                        node = regex.group(3).split(":")[0]
                        if node in self.data:
                            self.data[node] = [self.data[node][0] + float(pop_time), self.data[node][1] + 1]
                        else:
                            self.data[node] = [float(pop_time), 1]

                    progress.update()

    def print_stats(self):
        # Parse our log file if we haven't done that yet
        self.parse_log()

        total = sum(v[1] for k, v in self.data.items())
        print(f"Average node pop times (out of {total} pops in total)")
        for k, v in self.data.items():
            print(f"{k.split(':')[0]:15} {round(v[0]/v[1], 2)} secs {v[1]:-8} jobs from this node")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate local worker job pop statistics")
    parser.add_argument("-t", "--today", help="Statistics for today only", action="store_true")
    parser.add_argument("-y", "--yesterday", help="Statistics for yesterday only", action="store_true")
    parser.add_argument("-1", "--hour", help="Statistics for last hour only", action="store_true")
    args = vars(parser.parse_args())

    period = PERIOD_ALL
    if args["today"]:
        period = PERIOD_TODAY
    elif args["yesterday"]:
        period = PERIOD_YESTERDAY
    elif args["hour"]:
        period = PERIOD_HOUR

    logs = LogStats(period)
    logs.print_stats()
