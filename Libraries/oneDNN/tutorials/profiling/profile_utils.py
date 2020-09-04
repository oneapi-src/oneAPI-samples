#! /usr/bin/env python
import os, sys
import subprocess

os.environ['DNNL_VERBOSE'] = '1'
import psutil

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
        cpu_socket_count = int(subprocess.check_output(
            'cat /proc/cpuinfo | grep "physical id" | sort -u | wc -l', shell=True))
        print("Socket Number:", cpu_socket_count)
        print("=" * 20, "Memory Information", "=" * 20)
        # get the memory details
        svmem = psutil.virtual_memory()
        print("Total: ", int(svmem.total / (1024 ** 3)), "GB")
        self.cpufreq = cpufreq
        self.cpu_socket_count = cpu_socket_count
        self.svmem = svmem


def run_workload(outfile='mkldnn_log.csv'):
    print('Executing:', sys.argv[1:])
    output = subprocess.getoutput(' '.join(sys.argv[1:]))

    #print('Output:', output)

    with open(outfile, 'w') as f:
        for l in output.split('\n'):
            if 'dnnl' in l and 'exec' in l:
                f.write(l + '\n')

class oneDNNLog:

    def __init_(self):
        self.filename = ''
        self.data = None
        self.exec_data = None
        return

    def load_log(self, log):
        self.filename = log

        data = self.load_log_dnnl(log)
        count = data['time'].count()

        if count == 0:
            data = self.load_log_mkldnn(log)
            count = data['time'].count()

        exec_data = data[data['exec'] == 'exec']
        self.data = data
        self.exec_data = exec_data
        return

    def load_log_dnnl(self, log):
        import pandas as pd
        # dnnl_verbose,exec,cpu,convolution,jit:avx2,forward_inference,src_f32::blocked:abcd:f0 wei_f32::blocked:Acdb8a:f0 bia_f32::blocked:a:f0 dst_f32::blocked:aBcd8b:f0,,alg:convolution_direct,mb1_ic3oc96_ih227oh55kh11sh4dh0ph0_iw227ow55kw11sw4dw0pw0,1.21704
        data = pd.read_csv(log, names=[ 'dnnl_verbose','exec','arch','type', 'jit', 'pass', 'fmt', 'opt', 'alg', 'shape', 'time'])
        return data

    def load_log_mkldnn(self, log):
        import pandas as pd
        #mkldnn_verbose,exec,convolution,jit:avx512_common,forward_training,fsrc:nChw16c fwei:OIhw16i16o fbia:undef fdst:nChw16c,alg:convolution_direct,mb100_ic128oc32_ih7oh7kh3sh1dh0ph1_iw7ow7kw3sw1dw0pw1,0.201904
        print("load_log_mkldnn")
        data = pd.read_csv(log, names=[ 'mkldnn_verbose','exec','type', 'jit', 'pass', 'fmt', 'alg', 'shape', 'time'])
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
            jitstat['run2/run1'] = jitstat.iloc[:, 1] / jitstat.iloc[:, 0]
            jitstat_count = jitstat.sort_values('1-' + log1, ascending=False).head(n)
            print(jitstat_count)
        elif Type == "time":
            jitstat = pd.concat((d1.groupby(name)['time'].sum(), d2.groupby(name)['time'].sum()), axis=1, sort=True)
            jitstat.columns = ('1-' + log1, '2-' + log2)
            jitstat['run2/run1'] = jitstat.iloc[:, 1] / jitstat.iloc[:, 0]
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



if __name__ == '__main__':
    onednn = oneDNNUtils()
    if len(sys.argv) > 2 and '.csv' in sys.argv[1] and '.csv' in sys.argv[2]:
        log1 = oneDNNLog()
        log1.load_log(sys.argv[1])
        log2 = oneDNNLog()
        log2.load_log(sys.argv[2])
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
        print('Total MKLDNN time:', log.data['time'].sum())
        print('Total MKLDNN ops:', log.data['time'].count())
        onednn.breakdown(log.exec_data,"type","time")
        onednn.breakdown(log.exec_data,"jit","time")
    elif len(sys.argv) > 1:
        keyword = "_verbose"
        csvpath = onednn.parse_raw_output_to_csv(sys.argv[1], keyword=keyword)
        print(csvpath)
        log = oneDNNLog()
        log.load_log(csvpath)
        print('Total MKLDNN time:', log.data['time'].sum())
        print('Total MKLDNN ops:', log.data['time'].count())
        onednn.breakdown(log.exec_data,"type","time")
        onednn.breakdown(log.exec_data,"jit","time")
