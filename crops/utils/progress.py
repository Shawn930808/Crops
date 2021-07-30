import sys
import time
import os

class ProgressMonitor():
    def __init__(self):
        rows, columns = os.popen('stty size', 'r').read().split()
        self.width = int(columns) - 35
        self.current_time = None
        self.ticks = 0
        self.total_time = 0.0

    def _format_time(self, s):
        if (s > 1):
            return "{0:d}s{1:d}ms".format(int(s),int((s % 1) * 1000)) 
        else:
            return "{0:d}ms".format(int(s * 1000))

    def reset(self):
        self.current_time = None
        self.ticks = 0
        self.total_time = 0.0

    def update(self, count, total):
        if self.current_time == None:
            self.current_time = time.time()
            elapsed = None
            average = None
            remaining = None
        else:
            now = time.time()
            elapsed = now - self.current_time
            self.current_time = now
            self.total_time += elapsed
            self.ticks += 1
            average = self.total_time / self.ticks
            remaining = (total - count) * average

        completed = count == total

        bar_len = self.width
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        
        if completed:
            bar = '=' * (filled_len)
        else:
            bar = '=' * (filled_len - 1) + '>' + '.' * (bar_len - filled_len)

        label =  " {0}/{1} ".format(count, total)
        label_position = (bar_len // 2) - (len(label) // 2)
        bar[label_position:]
        bar = bar[:label_position] + label + bar[label_position+len(label):]

        if completed:
            sys.stdout.write(' [%s] Tot: %s | Step: %s   \r\n' % (bar, self._format_time(self.total_time), self._format_time(elapsed)))
        elif elapsed is not None and average is not None:
            sys.stdout.write(' [%s] ETA: %s | Step: %s   \r' % (bar, self._format_time(remaining), self._format_time(elapsed)))
        else:
            sys.stdout.write(' [%s] ETA: -- | Step: --   \r' % (bar))
        sys.stdout.flush()
