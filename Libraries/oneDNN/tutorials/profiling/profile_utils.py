#! /usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os, sys

os.environ['DNNL_VERBOSE'] = '1'
os.environ['DNNL_VERBOSE_TIMESTAMP'] = '1'
import psutil

import json
import time
import threading
from contextlib import contextmanager

try:
  from queue import Queue
except ImportError:
  from Queue import Queue


def to_microseconds(s):
  return 1000000 * float(s)

class PlatformUtils:

    def __init_(self):
        self.cpufreq = ''
        self.cpu_socket_count = ''
        self.svmem = ''
        return

    def dump_platform_info(self):
        # let's print CPU information
        print("=" * 20, "CPU Info", "=" * 20)
        # number of cores
        print("Physical cores:", psutil.cpu_count(logical=False))
        print("Total cores:", psutil.cpu_count(logical=True))
        # CPU frequencies
        cpufreq = psutil.cpu_freq()
        print("Max Frequency:", cpufreq.max)
        print("Min Frequency:", cpufreq.min)
        #cpu_socket_count = int(subprocess.check_output(
        #    'cat /proc/cpuinfo | grep "physical id" | sort -u | wc -l'))
        #print("Socket Number:", cpu_socket_count)
        print("=" * 20, "Memory Information", "=" * 20)
        # get the memory details
        svmem = psutil.virtual_memory()
        print("Total: ", int(svmem.total / (1024 ** 3)), "GB")
        self.cpufreq = cpufreq
        #self.cpu_socket_count = cpu_socket_count
        self.svmem = svmem


class FileUtils:

    def __init_(self):
        return
    def replace_string_in_file(self, filename, oldstring, newstring):
        fin = open(filename, "rt")
        #output file to write the result to
        fout = open('tmp.txt', "wt")
        #for each line in the input file
        for line in fin:
            #read replace the string and write to output file
            fout.write(line.replace(oldstring, newstring))
        #close input and output files
        fin.close()
        fout.close()
        os.remove(filename)
        os.rename('tmp.txt',filename)


class oneDNNLog:

    def __init_(self):
        self.filename = ''
        self.data = None
        self.exec_data = None
        self.with_timestamp = True
        return

    def load_log(self, log):
        self.filename = log
        self.with_timestamp = True
        data = self.load_log_dnnl_timestamp(log)
        count = data['time'].count()

        if count <= 1:
            data = self.load_log_dnnl(log)
            count = data['time'].count()
            self.with_timestamp = False

        if count == 0:
            data = self.load_log_mkldnn(log)
            count = data['time'].count()
            self.with_timestamp = False

        exec_data = data[data['exec'] == 'exec']
        self.data = data
        self.exec_data = exec_data.copy()

        if self.with_timestamp is True:
            import io
            with io.open('./oneDNN.json', mode='wb') as fh:
                tp = TraceProfiler(output=fh)
                tp.install()
                for index, row in self.data.iterrows():
                    if row["time"] != None and row["time"].find('.') != -1:
                        tp.fire_event(
                            event_type='exec',
                            event_name=row["type"],
                            event_cat='DNNL_Op',
                            kernel_name=row["jit"],
                            timestamp=str(float(row["timestamp"])*1000),
                            duration=str(float(row["time"])*1000),
                            pass_type=row["pass"],
                        )
                tp.shutdown()
        return

    def load_log_dnnl(self, log):
        import pandas as pd
        # dnnl_verbose,exec,cpu,convolution,jit:avx2,forward_inference,src_f32::blocked:abcd:f0 wei_f32::blocked:Acdb8a:f0 bia_f32::blocked:a:f0 dst_f32::blocked:aBcd8b:f0,,alg:convolution_direct,mb1_ic3oc96_ih227oh55kh11sh4dh0ph0_iw227ow55kw11sw4dw0pw0,1.21704
        data = pd.read_csv(log, names=[ 'dnnl_verbose','exec','arch','type', 'jit', 'pass', 'fmt', 'opt', 'alg', 'shape', 'time', 'dummy'], engine='python')
        return data

    def load_log_dnnl_timestamp(self, log):
        import pandas as pd
        # dnnl_verbose,629411020589.218018,exec,cpu,convolution,jit:avx2,forward_inference,src_f32::blocked:abcd:f0 wei_f32::blocked:Acdb8a:f0 bia_f32::blocked:a:f0 dst_f32::blocked:aBcd8b:f0,,alg:convolution_direct,mb1_ic3oc96_ih227oh55kh11sh4dh0ph0_iw227ow55kw11sw4dw0pw0,1.21704
        data = pd.read_csv(log, names=[ 'dnnl_verbose','timestamp','exec','arch','type', 'jit', 'pass', 'fmt', 'opt', 'alg', 'shape', 'time', 'dummy'], engine='python')
        return data

    def load_log_mkldnn(self, log):
        import pandas as pd
        #mkldnn_verbose,exec,convolution,jit:avx512_common,forward_training,fsrc:nChw16c fwei:OIhw16i16o fbia:undef fdst:nChw16c,alg:convolution_direct,mb100_ic128oc32_ih7oh7kh3sh1dh0ph1_iw7ow7kw3sw1dw0pw1,0.201904
        print("load_log_mkldnn")
        data = pd.read_csv(log, names=[ 'mkldnn_verbose','exec','type', 'jit', 'pass', 'fmt', 'alg', 'shape', 'time'], engine='python')
        return data


class oneDNNUtils:

    def __init_(self):
        self.topk=50
        self.logx=True 
        self.figsize=(10,10)
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(18, 15))
        self.ax = fig.add_subplot(111)
        return

    def breakdown(self, data, Group, Type):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(18, 15))
        ax = fig.add_subplot(111)
        figsize=(10,10)
        topk=50
        try:
            if Type == "time":
                print()
                print(' breakdown:',Group)
                data['time'] = data['time'].astype(float)
                time = data.groupby(Group)['time'].sum().sort_values().head(topk)
                print(time)
                title=Group + "Time Breakdown"
                time[:topk].plot.pie(
                    ax=ax, title=title, figsize=figsize, logx=True, autopct='%1.1f%%')
                ax.figure.savefig(title)
            elif Type == "count":
                print()
                count = data[Group].value_counts().head(topk)
                print(count)
                title=Group+"Count Breakdown"
                count[:topk].plot.bar(
                    ax=ax, title=title, figsize=figsize, logx=False, rot=45)
                ax.figure.savefig(title)
        except:
            print("Exception!")
            pass
        return

    def stats_comp(self, name, Type,onednn_log1, onednn_log2, n=50):
        import pandas as pd
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(18, 15))
        ax = fig.add_subplot(111)
        figsize=(10,10)
        topk=50

        d1 = onednn_log1.exec_data
        log1 = onednn_log1.filename
        d2 = onednn_log2.exec_data
        log2 = onednn_log2.filename
        print(name, 'stats:')
        if Type == "count":
            jitstat = pd.concat((d1[name].value_counts(), d2[name].value_counts()), axis=1, sort=True)
            jitstat.columns = ('1-' + log1, '2-' + log2)
            jitstat['run2/run1'] = jitstat.iloc[:, 1].astype(float) / jitstat.iloc[:, 0].astype(float)
            jitstat_count = jitstat.sort_values('1-' + log1, ascending=False).head(n)
            print(jitstat_count)
        elif Type == "time":
            d1['time'] = d1['time'].astype(float)
            d2['time'] = d2['time'].astype(float)
            jitstat = pd.concat((d1.groupby(name)['time'].sum(), d2.groupby(name)['time'].sum()), axis=1, sort=True)
            jitstat.columns = ('1-' + log1, '2-' + log2)
            jitstat['run2/run1'] = jitstat.iloc[:, 1].astype(float) / jitstat.iloc[:, 0].astype(float)
            jitstat_time = jitstat.sort_values('1-' + log1, ascending=False).head(n)
            print(jitstat_time)
            title=name + " run2/run1 Time Comparison"
            jitstat_compare = jitstat_time.drop(columns=['1-' + log1, '2-' + log2])
            if len(jitstat_compare) == 0:
                return
            jitstat_compare[:topk].plot.bar(
                ax=ax, title=title, figsize=figsize, logx=False, rot=45)
            filename = name + " Time Comparison"
            ax.figure.savefig(filename)
    def parse_raw_output_to_csv(self, filepath, csvpath='mkldnn_log.csv', keyword='dnnl_verbose'):
        #filepath = 'Iliad.txt'
        import csv

        with open(csvpath, "w") as file:
            with open(filepath) as fp:
                line = fp.readline()
                cnt = 1
                while line:
                    if line.find(keyword) != -1:
                        file.write(line)
                        #print("Line {}: {}".format(cnt, line.strip()))
                    line = fp.readline()
                    cnt += 1
        return csvpath

class TraceWriter(threading.Thread):

  def __init__(self, terminator, input_queue, output_stream):
    threading.Thread.__init__(self)
    self.daemon = True
    self.terminator = terminator
    self.input = input_queue
    self.output = output_stream

  def _open_collection(self):
    """Write the opening of a JSON array to the output."""
    self.output.write(b'[')

  def _close_collection(self):
    """Write the closing of a JSON array to the output."""
    self.output.write(b'{}]')  # empty {} so the final entry doesn't end with a comma

  def run(self):
    self._open_collection()
    while not self.terminator.is_set() or not self.input.empty():
      item = self.input.get()
      self.output.write((json.dumps(item) + ',\n').encode('ascii'))
    self._close_collection()


class TraceProfiler(object):
  """A python trace profiler that outputs Chrome Trace-Viewer format (about://tracing).

     Usage:

        from pytracing import TraceProfiler
        tp = TraceProfiler(output=open('/tmp/trace.out', 'wb'))
        with tp.traced():
          ...

  """
  TYPES = {'call': 'B', 'return': 'E', 'exec': 'X'}

  def __init__(self, output, clock=None):
    self.output = output
    self.clock = clock or time.time
    self.pid = os.getpid()
    self.queue = Queue()
    self.terminator = threading.Event()
    self.writer = TraceWriter(self.terminator, self.queue, self.output)

  @property
  def thread_id(self):
    return threading.current_thread().name

  @contextmanager
  def traced(self):
    """Context manager for install/shutdown in a with block."""
    self.install()
    try:
      yield
    finally:
      self.shutdown()

  def install(self):
    """Install the trace function and open the JSON output stream."""
    self.writer.start()               # Start the writer thread.

  def shutdown(self):
    self.terminator.set()              # Stop the writer thread.
    self.writer.join()                 # Join the writer thread.

  def fire_event(self, event_type, event_name, event_cat, kernel_name, timestamp, duration, pass_type):
    """Write a trace event to the output stream."""
    # https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview
    event = dict(
      name=event_name,                 # Event Name.
      cat=event_cat,               # Event Category.
      tid=self.thread_id,             # Thread ID.
      ph=self.TYPES[event_type],      # Event Type.
      pid=self.pid,                   # Process ID.
      ts=timestamp,                   # Timestamp.
      dur=duration,
      args=dict(
        jit=kernel_name,
        pass_type=pass_type,
        )
      )
    self.queue.put(event)



if __name__ == '__main__':
    onednn = oneDNNUtils()
    if len(sys.argv) > 2 and '.csv' in sys.argv[1] and '.csv' in sys.argv[2]:
        log1 = oneDNNLog()
        log1.load_log(sys.argv[1])
        log2 = oneDNNLog()
        log2.load_log(sys.argv[2])
        log1.data['time'] = log1.data['time'].astype(float)
        print('Total time %s: %0.2f\t---  %s: %0.2f' % (log1.filename, log1.data['time'].sum(), log2.filename, log2.data['time'].sum()))
        print('Total  ops  %s: %d\t\t---  %s: %d'    % (log1.filename, log1.data['time'].count(), log2.filename, log2.data['time'].count()))
        #onednn.stats_comp('jit', 'time',log1, log2)

        print()
        onednn.stats_comp('type', 'time',log1, log2)

        #print()
        #onednn.stats_comp('shape', 'time',log1, log2)

    elif len(sys.argv) > 1 and '.csv' in sys.argv[1]:
        log = oneDNNLog()
        log.load_log(sys.argv[1])
        log.exec_data['time'] = log.exec_data['time'].astype(float)
        print('Total MKLDNN time:', log.exec_data['time'].sum())
        print('Total MKLDNN ops:', log.exec_data['time'].count())
        onednn.breakdown(log.exec_data,"type","time")
        onednn.breakdown(log.exec_data,"jit","time")
    elif len(sys.argv) > 1:
        keyword = "_verbose"
        csvpath = onednn.parse_raw_output_to_csv(sys.argv[1], keyword=keyword)
        print(csvpath)
        log = oneDNNLog()
        log.load_log(csvpath)
        log.exec_data['time'] = log.exec_data['time'].astype(float)
        print('Total MKLDNN time:', log.exec_data['time'].sum())
        print('Total MKLDNN ops:', log.exec_data['time'].count())
        onednn.breakdown(log.exec_data,"type","time")
        onednn.breakdown(log.exec_data,"jit","time")
