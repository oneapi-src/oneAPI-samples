import sys
import configparser
import tensorflow as tf
import json
from tensorflow.python.client import timeline
import os, fnmatch
import psutil
import ast
import tensorflow.estimator
from tensorflow.python.training import training_util
try:
    from git import Repo
    has_git = True
except ImportError as e:
    print(e)
    print("can't import git module")
    has_git = False
    pass

def do_command(command):
    print("WARNING!! No execution for : ", command)

class TimeLiner:

    def __init__(self):
        self._timeline_dict = None

    def update_timeline(self, chrome_trace):
        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            self._timeline_dict['traceEvents'] += [event for event in chrome_trace_dict['traceEvents'] if 'ts' in event]

    def update_timeline_from_runmeta(self, run_metadata):
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        self.update_timeline(chrome_trace)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)


class tfSession(tf.compat.v1.Session):

    def __init__(self, target='', graph=None, config=None, run_metadata=None, many_runs_timeline=None):
        self.run_metadata = run_metadata
        self.options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        self.many_runs_timeline = many_runs_timeline
        super(tfSession, self).__init__(target, graph=graph, config=config)

    def run(self, c, feed_dict=None, options=None, run_metadata=None):
        options = self.options
        run_metadata = self.run_metadata
        ret = super(tfSession, self).run(c, feed_dict=feed_dict, options=options, run_metadata=run_metadata)
        self.many_runs_timeline.update_timeline_from_runmeta(run_metadata)
        return ret

    def save_timeline(self, fname):
        self.many_runs_timeline.save(fname)


class tfProfileHook(tf.estimator.ProfilerHook):
    def __init__(self, save_steps=None, save_secs=None, output_dir="", json_fname="", timeline_count=10):
        self._output_tag = "blah-{}"
        self._output_dir = output_dir
        self._timer = tf.estimator.SecondOrStepTimer(every_secs=save_secs,
                                                     every_steps=save_steps)
        self._atomic_counter = 0
        self.many_runs_timeline = TimeLiner()
        self.timeline_count = timeline_count
        import os
        ProfileUtilsRoot = os.environ['ProfileUtilsRoot']
        self.json_fname = ProfileUtilsRoot + "/../" + json_fname
        if output_dir == "":
            output_dir = ProfileUtilsRoot + "/../"

    def begin(self):
        self._next_step = None
        self._global_step_tensor = training_util.get_global_step()

        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use ProfilerHook.")

    def before_run(self, run_context):
        self._request_summary = (self._next_step is None or self._timer.should_trigger_for_step(self._next_step))
        requests = {}
        opts = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        return tf.estimator.SessionRunArgs(requests, options=opts)

    def after_run(self, run_context, run_values):

        global_step = self._atomic_counter + 1
        self._atomic_counter = self._atomic_counter + 1
        self._next_step = global_step + 1

        self.many_runs_timeline.update_timeline_from_runmeta(run_values.run_metadata)
        if self._atomic_counter == self.timeline_count:
            self.many_runs_timeline.save(self.json_fname)

    def end(self, session):
        if self._atomic_counter < self.timeline_count:
            self.many_runs_timeline.save(self.json_fname)


class TensorflowUtils:

    def is_mkl_enabled(self):
        major_version = int(tf.__version__.split(".")[0])
        if major_version >= 2:
            from tensorflow.python import _pywrap_util_port
            return _pywrap_util_port.IsMklEnabled()
        else:
            return tf.pywrap_tensorflow.IsMklEnabled()


class GitOps:

    def __init__(self, repopath='./'):
        self.repopath = repopath
        if has_git is True:
            self.repo = Repo(repopath)
        return

    def find_commit_with_keyword(self, keyword, search_commits_depth=5):
        ret = False
        repo = Repo(self.repopath)
        if has_git is False:
            return ret
        if not repo.bare:
            try:
                commits = list(repo.iter_commits())[:search_commits_depth]
                for commit in commits:
                    if commit.summary.find(keyword) != -1:
                        ret = True
                        return ret
            except:
                print("EXCEPTION : Find commit %s ", keyword)
                pass
        return ret


class PlatformUtils:

    def __init_(self):
        self.cpufreq = ''
        self.cpu_socket_count = ''
        self.svmem = ''
        return

    def dump_platform_info(self):
        # Import platform_util and print CPU information
        if 'ModelZooRoot' in os.environ and os.environ['ModelZooRoot']:
            zoo_root = os.environ['ModelZooRoot']
            platform_util = os.path.join(zoo_root, 'benchmarks/common')
        else:
            file_dir = os.path.dirname(os.path.abspath(__file__))
            platform_util = os.path.join(file_dir, '../../../../benchmarks/common')
        sys.path.insert(0, platform_util)
        import platform_util
        cpu_info = platform_util.CPUInfo()
        print("=" * 20, "CPU Info", "=" * 20)
        # number of cores
        print("Physical cores per socket:", cpu_info.cores_per_socket)
        print("Total physical cores:", cpu_info.cores)
        # CPU frequencies
        cpufreq = psutil.cpu_freq()
        print("Max Frequency:", cpufreq.max)
        print("Min Frequency:", cpufreq.min)
        print("Socket Number:", cpu_info.sockets)
        print("=" * 20, "Memory Information", "=" * 20)
        # get the memory details
        svmem = psutil.virtual_memory()
        print("Total: ", int(svmem.total / (1024 ** 3)), "GB")
        self.cpufreq = cpufreq
        self.cpu_socket_count = cpu_info.sockets
        self.svmem = svmem


class CommonUtils:

    def __init__(self):
        return

    def found_files_in_folder(self, pattern, path):
        listOfFiles = os.listdir(path)
        foundfiles = []
        founpaths = []
        for f in listOfFiles:
            if fnmatch.fnmatch(f, pattern):
                foundfiles.append(f)
                founpaths.append( path+os.sep+f)
        return foundfiles, founpaths

    def found_files_in_folders(self, pattern, paths):
        foundfiles = []
        foundpaths = []
        for path in paths:
            files ,paths = self.found_files_in_folder(pattern, path)
            foundfiles += files
            foundpaths += paths
        return foundfiles, foundpaths


class ConfigFile:

    def __init__(self, confpath='profiling/topo.ini'):
        self.configpath = confpath
        self.wget = ''
        self.data_download = ''
        self.data_location = ''
        self.preprocessing = ''
        self.in_graph = ''
        self.checkpoint = ''
        self.tf_util = TensorflowUtils()
        self.json_fname = ''
        if self.tf_util.is_mkl_enabled() is True:
            self.json_fname = 'mkl_'
        else:
            self.json_fname = 'stock_'
        self.patches = ''
        self.patched = False
        self.patches_keyword = ''
        self.throughput_keyword = ''
        self.throughput_index = -1
        self.mkl_only = True
        self.support_accuracy = False

    def read_section(self):
        config = configparser.ConfigParser()
        config.read(self.configpath)
        return config.sections()

    def read_supported_section(self, on_mkl=False, accuracy_only=False):
        config = configparser.ConfigParser()
        config.read(self.configpath)
        supported_sections = []
        for each_section in config.sections():
            is_supported = True
            for each_key, each_val in config.items(each_section):
                if each_key == 'mkl-only':
                    if each_val is not None:
                        if ast.literal_eval(each_val) is True and on_mkl is False:
                            is_supported = False
                if each_key == 'support-accuracy':
                    if each_val is not None:
                        if ast.literal_eval(each_val) is False and accuracy_only is True:
                            is_supported = False
            if is_supported is True:
                supported_sections.append(each_section)

        return supported_sections

    def convert_configs_to_pd_dataframe(self):

        config = configparser.ConfigParser()
        config.read(self.configpath)
        import pandas as pd
        import numpy as np

        index_list = []
        data_list = []
        columns_list = ['benchmark','model-name', 'mode', 'precision','patches','json-fname']
        for section in config.sections():
            index_list.append(section)
            data = []
            data.append(section)
            data.append(config.get(section, columns_list[1]))
            data.append(config.get(section, columns_list[2]))
            data.append(config.get(section, columns_list[3]))
            data.append(config.get(section, columns_list[4]))
            data.append(config.get(section, columns_list[5]))
            data_list.append(data)
        df = pd.DataFrame(data_list, columns = columns_list)

        df_types = df.groupby([ columns_list[1],  columns_list[2]]).filter(lambda x: len(x) >= 2)

        df_types_obj = df_types.groupby([ columns_list[1],  columns_list[2]])

        return df, df_types, df_types_obj

    def read_value_from_section(self, topo_name, key):
        config = configparser.ConfigParser()
        config.read(self.configpath)
        string_val = config.get(topo_name, key)
        return string_val

    def write_value_from_section(self, topo_name, key, val):
        config = configparser.ConfigParser()
        config.read(self.configpath)
        config.set(topo_name, key, val)

        # save to a file
        with open(self.configpath, 'w') as configfile:
             config.write(configfile)
        return

    def read_config(self, topo_name):
        configs = []
        config = configparser.ConfigParser()
        config.read(self.configpath)
        for each_section in config.sections():
            if each_section == topo_name:
                for each_key, each_val in config.items(each_section):
                    key = '--' + each_key
                    #if each_key == 'data-location':
                    #    if each_val is not None:
                    #        self.benchmark_only = False
                    if each_key == 'throughput-keyword':
                        if each_val is not None:
                            self.throughput_keyword = each_val
                    elif each_key == 'throughput-index':
                        if each_val is not None:
                            self.throughput_index = each_val
                    elif each_key == 'wget':
                        if each_val is not None:
                            self.wget = each_val
                    elif each_key == 'data-download':
                        if each_val is not None:
                            self.data_download = each_val
                    elif each_key == 'preprocessing':
                        if each_val is not None:
                            self.preprocessing = each_val
                    elif each_key == 'data-location':
                        if each_val is not None:
                            if each_val != '':
                                print("data-location : ", each_val)
                                configs.append(key)
                                configs.append(each_val)
                                self.data_location = each_val
                    elif each_key == 'in-graph':
                        if each_val != '':
                            configs.append(key)
                            configs.append(each_val)
                            self.in_graph = each_val
                        self.checkpoint = 'NA'
                    elif each_key == 'checkpoint':
                        if each_val != '':
                            configs.append(key)
                            configs.append(each_val)
                            self.checkpoint = each_val
                        self.in_graph = 'NA'
                    elif each_key == 'json-fname':
                        if each_val is not None:
                            self.json_fname = self.json_fname + each_val
                    elif each_key == 'patches':
                        if each_val is not None:
                            self.patches = each_val
                            val = each_val.split('.')[0]
                            val = val.split('-')[1:]
                            keyword = " ".join(val)
                            self.patches_keyword = keyword
                    elif each_key == 'mkl-only':
                        if len(each_val) != 0 :
                            self.mkl_only = ast.literal_eval(each_val)
                    elif each_key == 'support-accuracy':
                        if len(each_val) != 0 :
                            self.support_accuracy = ast.literal_eval(each_val)
                    else:
                        if len(each_val) != 0 :
                            if each_val[0] == '=':
                                configs.append(key+each_val)
                            else:
                                configs.append(key)
                                configs.append(each_val)

        return configs

    def get_parameters(
            self, topo_name, configvals, batch_size=1, thread_number=1,
            socket_number=1, num_inter_threads=0, num_intra_threads=0, accuracy_only=False):
        benchmark_argvs = []
        benchmark_argvs = benchmark_argvs + configvals
        benchmark_argvs.append('--framework')
        benchmark_argvs.append('tensorflow')
        if batch_size > 0:
            benchmark_argvs.append('--batch-size')
            benchmark_argvs.append(str(batch_size))
        if accuracy_only is True:
            benchmark_argvs.append('--accuracy-only')
        else:
            benchmark_argvs.append('--benchmark-only')

        if num_inter_threads > 0:
            benchmark_argvs.append('--num_inter_threads=' + str(num_inter_threads))
            benchmark_argvs.append('--data_num_inter_threads=' + str(num_inter_threads))
        else:
            benchmark_argvs.append('--num_inter_threads=' + str(socket_number))
            benchmark_argvs.append('--data_num_inter_threads=' + str(socket_number))

        if num_intra_threads > 0:
            benchmark_argvs.append('--num_intra_threads=' + str(num_intra_threads))
            benchmark_argvs.append('--data_num_intra_threads=' + str(num_intra_threads))
        else:
            benchmark_argvs.append('--num_intra_threads=' + str(thread_number))
            benchmark_argvs.append('--data_num_intra_threads=' + str(thread_number))

        if thread_number > 0:
            benchmark_argvs.append('--num-cores=' + str(thread_number))
        if socket_number == 1:
            benchmark_argvs.append('--socket-id')
            benchmark_argvs.append('0')

        return benchmark_argvs
   
    def uncompress_file(self, filepath, pretrainfd='pretrained', current_path='./'):
        import shutil
        uncompress_path = filepath
        full_filename = filepath.split(os.sep)[-1]

        file_ext = full_filename.split('.')[-1]
        filename = full_filename.split('.')[0]
        cmd = ''
        if file_ext == 'zip':
            cmd = "unzip " + filepath
        elif file_ext == 'gz':
            cmd = "tar -xzvf  " + filepath
        if cmd != '':
            do_command(cmd)
            if os.path.exists(pretrainfd + os.sep + filename) is False:
                shutil.move(filename, pretrainfd)
            uncompress_path = filepath.split('.')[0]

        return uncompress_path

    def download_dataset(self, datasetfd='dataset', current_path='./'):
        import shutil
        cmd = self.data_download
        filename = self.wget.split('/')[-1]
        dataset_path = current_path + os.sep + datasetfd
        if os.path.exists(dataset_path) is True:
            return dataset_path
        do_command(cmd)
        print('Downloaded the model in:', dataset_path)
        return dataset_path

    def download_pretrained_model(self, pretrainfd='pretrained', current_path='./'):
        import shutil
        cmd = "wget " + self.wget
        filename = self.wget.split('/')[-1]
        pretrain_model_path = current_path + os.sep + pretrainfd + os.sep + filename
        if os.path.exists(pretrain_model_path) is True:
            return pretrain_model_path
        do_command(cmd)
        if os.path.exists(pretrainfd) is False:
            os.mkdir(pretrainfd)
        if os.path.exists(pretrainfd + os.sep + filename) is False:
            shutil.move(filename, pretrainfd)
        print('Downloaded the model in:', pretrain_model_path)
        return pretrain_model_path

    def patch_model_to_enable_timeline(self, patch_fd="profiling/patches", repopath="../../"):
        if self.patches == '':
            return
        self.git = GitOps(repopath=repopath)
        print("patch keyword: ", self.patches_keyword)
        if self.git.find_commit_with_keyword(self.patches_keyword) is True:
            self.patched = True
            print("has been patched, no action")
            return
        print(self.patches)
        patch_path = patch_fd + os.sep + self.patches
        if os.path.exists(patch_path) is True:
            cmd = "git am " + patch_path
            print(cmd)
            do_command(cmd)
            self.patched = True
        return

    def unpatch_model_to_enable_timeline(self, model_path='../../../models/'):
        if self.patched is False:
            print("has not been patched, no action")
            return
        print("unpatch keyword: ", self.patches_keyword)
        if self.git.find_commit_with_keyword(self.patches_keyword, search_commits_depth=2) is True:
            import os
            print("do unpatch")
            cmd = "git reset HEAD^ "
            print(cmd)
            do_command(cmd)
            cmd = "git checkout " + model_path + "*"
            print(cmd)
            do_command(cmd)
        return


class PerfPresenter:

    def __init__(self, showAbsNumber=False):
        self.tf_util = TensorflowUtils()
        self.showAbsNumber = showAbsNumber
        pass

    def autolabel(self, ax, rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                '{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom')

    def draw_perf_diag(self, topo_name, a_means, b_means, a_label, b_label):
        import matplotlib.pyplot as plt
        import numpy as np
        labels = [topo_name]
        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, a_means, width, label=a_label)
        rects2 = ax.bar(x + width / 2, b_means, width, label=b_label)
        ax.set_ylabel('Throughput')
        ax.set_title('stock TF vs Intel TF')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        self.autolabel(ax, rects1)
        self.autolabel(ax, rects2)
        fig.tight_layout()
        plt.show()

    def draw_perf_ratio_diag(self, topo_name, a_means, b_means, a_label, b_label):
        import matplotlib.pyplot as plt
        import numpy as np
        labels = [topo_name]
        a_ratio = 1
        b_ratio = float(b_means / a_means)
        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, a_ratio, width, label=a_label)
        rects2 = ax.bar(x + width / 2, b_ratio, width, label=b_label)
        ax.set_ylabel('Throughput Speeup ')
        ax.set_title('stock TF vs Intel TF')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        self.autolabel(ax, rects1)
        self.autolabel(ax, rects2)
        fig.tight_layout()
        plt.show()

    def create_csv_logfile(self, Type, filename):
        import csv
        import os.path
        if Type == 'training':
            fnames = ['mkl', 'elapsed_time']
        else:
            fnames = ['mkl', 'elapsed_time', 'throughput', 'accuracy']
        if os.path.isfile(filename):
            print('file exists')
        else:
            f = open(filename, 'w')
            with f:
                writer = csv.DictWriter(f, fieldnames=fnames)
                writer.writeheader()

    def get_diff_from_csv_filenames(self, x, y):
        x_split=x.split('_')
        y_split=y.split('_')
        if len(x_split) != len(y_split):
            print("ERROR! can't two files have different formats")
            return '',''
        for i in range(len(x_split)):
            if x_split[i] !=  y_split[i]:
                break
        return x_split[i], y_split[i]

    def log_infer_perfcsv(self, elapsed_time, throughput, accuracy ,filename):
        import csv
        f = open(filename, 'a')
        with f:
            fnames = ['mkl', 'elapsed_time', 'throughput', 'accuracy']
            writer = csv.DictWriter(f, fieldnames=fnames)
            writer.writerow(
                {'mkl': self.tf_util.is_mkl_enabled(),
                 'elapsed_time': elapsed_time,
                 'throughput': throughput,
                 'accuracy': accuracy})

    def log_train_perfcsv(self, elapsed_time, filename):
        import csv
        f = open(filename, 'a')
        with f:
            fnames = ['mkl', 'elapsed_time']
            writer = csv.DictWriter(f, fieldnames=fnames)
            writer.writerow(
                {'mkl': self.tf_util.is_mkl_enabled(),
                 'elapsed_time': elapsed_time})

    def read_number_from_csv(self, filepath, framename):
        import csv
        import statistics
        f = open(filepath, 'r')
        col1 = []
        col2 = []
        mean1 = 0
        mean2 = 0
        stdev1 = 0
        stdev2 = 0
        with f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['mkl'] == 'True':
                    col1.append(float(row[framename]))
                else:
                    col2.append(float(row[framename]))

        if len(col1) > 0:
            mean1 = statistics.mean(col1)
        if len(col2) > 0:
            mean2 = statistics.mean(col2)
        if len(col1) > 1:
            stdev1 = statistics.stdev(col1)
        if len(col2) > 1:
            stdev2 = statistics.stdev(col2)
        return (mean1, mean2, len(col1), len(col2), stdev1, stdev2)

    def plot_perf_graph(self, ylabel, xlabel, stock_number, intel_number, stock_stdev, intel_stdev):
        import matplotlib.pyplot as plt
        import numpy as np
        labels = [xlabel]
        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, stock_number, width, yerr=stock_stdev, label='stock TF')
        rects2 = ax.bar(x + width / 2, intel_number, width, yerr=intel_stdev, label='intel TF')
        ax.set_ylabel(ylabel)
        ax.set_title('stock TF vs Intel TF')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    '{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center',
                    va='bottom')

        autolabel(rects1)
        autolabel(rects2)
        fig.tight_layout()
        plt.show()

    def plot_perf_graph_v2(self, ylabel, xlabel, means_list, stddev_list, filepath_list, label_list,title='stock TF vs Intel TF'):
        import matplotlib.pyplot as plt
        import numpy as np
        labels = [xlabel]
        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots()

        rects_list = []
        for i in range(len(filepath_list)):
            number = means_list[i]
            stddev = stddev_list[i]
            filepath = filepath_list[i]
            label = label_list[i]
            rects = ax.bar(x - width / 2 + width*i, number, width, yerr=stddev, label=label )
            rects_list.append(rects)

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    '{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center',
                    va='bottom')

        for i in range(len(rects_list)):
            autolabel(rects_list[i])
        fig.tight_layout()
        plt.show()

    def draw_perf_diag_from_csv(self, filepath, framename, y_axis_name, title):
        if self.showAbsNumber is False:
            return
        stock_means = []
        intel_means = []
        stock_stdev = []
        intel_stdev = []
        mean1, mean2, len1, len2, stdev1, stdev2 = self.read_number_from_csv(filepath, framename)
        stock_means.append(float(mean2))
        intel_means.append(float(mean1))
        stock_stdev.append(float(stdev2))
        intel_stdev.append(float(stdev1))
        self.plot_perf_graph(y_axis_name, title, stock_means, intel_means, stock_stdev, intel_stdev)

    def draw_perf_ratio_diag_from_csv(self, filepath, framename, y_axis_name, title):
        stock_ratio_means = [1]
        intel_ratio_means = []
        stock_stdev = []
        intel_stdev = []
        mean1, mean2, len1, len2, stdev1, stdev2 = self.read_number_from_csv(filepath, framename)
        if mean1 is 0 or mean2 is 0:
            print("ERROR. Users must run the benchmark with both Stock TF and Intel TF\n")
            return
        intel_ratio_means.append(float(mean1 / mean2))
        stock_stdev.append(float(stdev2 / mean2))
        intel_stdev.append(float(stdev1 / mean1))
        self.plot_perf_graph(y_axis_name, title, stock_ratio_means, intel_ratio_means, stock_stdev, intel_stdev)

    def draw_perf_diag_from_csvs(self, filepath_list,label_list ,framename, y_axis_name, x_axis_name, title, analyze_mkl=True):
        if self.showAbsNumber is False:
            return
        means_list = []
        stddev_list = []
        for filepath in filepath_list:
            means = []
            stdev = []
            mean1, mean2, len1, len2, stdev1, stdev2 = self.read_number_from_csv(filepath, framename)
            if analyze_mkl == True:
                means.append(float(mean1))
                stdev.append(float(stdev1))
            else:
                means.append(float(mean2))
                stdev.append(float(stdev2))
            means_list.append(means)
            stddev_list.append(stdev)
        self.plot_perf_graph_v2(y_axis_name, x_axis_name, means_list, stddev_list, filepath_list, label_list,title=title)

    def draw_perf_ratio_diag_from_csvs(self, filepath_list,label_list ,framename, y_axis_name, x_axis_name, title, analyze_mkl=True):
        means_list = []
        stddev_list = []
        dividend = 0
        for filepath in filepath_list:
            means = []
            stdev = []
            ratio_means = []
            ratio_stddev = []
            mean1, mean2, len1, len2, stdev1, stdev2 = self.read_number_from_csv(filepath, framename)
            if analyze_mkl == True:
                means.append(float(mean1))
                stdev.append(0)
                if dividend == 0:
                    dividend = mean1
                ratio_means.append(float(mean1 / dividend))
            else:
                means.append(float(mean2))
                stdev.append(0)
                if dividend == 0:
                    dividend = mean2
                ratio_means.append(float(mean2 / dividend))
            means_list.append(ratio_means)
            stddev_list.append(stdev)
        self.plot_perf_graph_v2(y_axis_name, x_axis_name, means_list, stddev_list, filepath_list, label_list,title=title)

    def parse_stdout(self, output, keyword):
        output = str(output)
        output = output.split("\n")
        lines = output[0].split('\\')
        for l in lines:
            if l.find("Throughput") > 0:
                return l
        return None

    def read_throughput(self, filepath, keyword='Throughput', index=1):
        number = None
        with open(filepath) as (fp):
            line = fp.readline()
            while line:
                line = fp.readline()
                if line.find(keyword) != -1:
                    print(line)
                    number = line.split(' ')[index]
                    print(number)
        return number

    def read_accuracy(self, filepath, keyword='accuracy', index=-2):
        accuracy=[]
        with open(filepath) as (fp):
            line = fp.readline()
            while line:
                line = fp.readline()
                if line.find(keyword) != -1:
                    number = line.split(' ')[index]
                    number = number.strip(',')
                    number = number.strip('(')
                    number = number.strip(')')
                    accuracy.append(float(number))
        return accuracy

    def read_iteration_time(self, filepath, keyword='Iteration', index=-2):
        iteration=[]
        with open(filepath) as (fp):
            line = fp.readline()
            while line:
                line = fp.readline()
                if line.find(keyword) != -1:
                    number = line.split(' ')[index]
                    number = number.strip(',')
                    number = number.strip('(')
                    number = number.strip(')')
                    iteration.append(float(number))
        return iteration


class TFTimelinePresenter:

    def __init__(self, showAbsNumber=False):
        self.showAbsNumber = showAbsNumber
        import pandas as pd
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1500)
        self.pd = pd

    def show(self, diag, Type):
        import matplotlib.pyplot as plt
        ret = None
        if self.showAbsNumber is True or Type == 'pie':
            ret = diag
            plt.show()
        plt.clf()
        return ret

    def read_timeline(self, fn, maxents=0):
        import json
        import pandas as pd
        with open(fn, 'r') as (f):
            j = json.loads(f.read())
        allkeys = [list(js.keys()) for js in j['traceEvents']]
        allkeys = [item for sublist in allkeys for item in iter(sublist)]
        allkeys = sorted(set(allkeys))
        argkeys = [js['args'].keys() for js in j['traceEvents'] if 'args' in js]
        argkeys = sorted(set([item for sublist in argkeys for item in iter(sublist)]))
        argkeys = ['arg_' + k for k in argkeys]
        entries = []
        for i, e in enumerate(j['traceEvents']):
            if maxents != 0 and i > maxents:
                break
            ent = {}
            for k, v in e.items():
                if k == 'args':
                    for a in v.keys():
                        ent['arg_' + a] = str(v[a])

                else:
                    ent[k] = str(v)

            entries.append(ent)

        df = pd.DataFrame(entries)
        return df

    def summarize_item(self, tl, item, ascending=False):
        return tl.groupby([item])['dur'].sum().sort_values(ascending=ascending)

    def summarize_barh(self, tl, item, topk=15, ascending=False, ax=None, title=None, figsize=None, logx=False):
        ret = self.summarize_item(tl, item, ascending)[:topk].plot.barh(
            ax=ax, title=title, figsize=figsize, logx=False)
        return ret

    def summarize_pie(self, tl, item, topk=15, ascending=False, ax=None, title=None, figsize=None, logx=False):
        ret = self.summarize_item(tl, item, ascending)[:topk].plot.pie(
            ax=ax, title=title, figsize=figsize, logx=logx, autopct='%1.1f%%')
        return ret

    def opname(self, x):
        return self.demangle(x.split('/')[(-1)])

    def opbase(self, x):
        return self.opname(x).split(':')[(-1)]

    def blockname(self, x):
        return x.split('/')[0].split('_')[0]

    def postprocess_timeline(self, t):
        t['dur'] = t['dur'].astype('float') / 1000
        t['ts'] = t['ts'].astype('float') - t['ts'].astype('float').min()
        t['arg_name'] = t['arg_name'].astype('str').replace('nan', '')
        t['arg_opname'] = t['arg_name'].apply(self.opname)
        t['arg_opbase'] = t['arg_name'].apply(self.opbase)
        return t

    def demangle(self, x, short=True):
        import cxxfilt
        z = x.split('::')
        try:
            if short:
                return z[0] + '::' + cxxfilt.demangle(z[1]).split('<')[0]
            else:
                return z[0] + '::' + cxxfilt.demangle(z[1])
        except IndexError:
            return x

    def get_tf_ops_time(self, timeline_pd, fn, tfile_prefix):
        sitems = self.summarize_item(timeline_pd, 'arg_op')
        import csv
        filename = fn.split('.')[0] + '.csv'
        f = open(filename, 'w')
        time_col_name = 'elapsed_time_' + tfile_prefix
        with f:
            fnames = ['op', time_col_name, 'speedup', 'mkl_op']
            writer = csv.DictWriter(f, fieldnames=fnames)
            writer.writeheader()
            x = 0
            for sitem in sitems:
                mkl_op = False
                if tfile_prefix == 'mkl':
                    op_name = sitems.index[x].strip('_')
                    if op_name.find('Mkl') != -1:
                        mkl_op = True
                        op_name = op_name[3:]
                else:
                    op_name = sitems.index[x].strip('_')
                writer.writerow({'op': op_name, time_col_name: sitems[x], 'speedup': 0, 'mkl_op': mkl_op})
                x = x + 1
        ret = None
        if self.showAbsNumber is True:
            ret = sitems
        return ret

    def plot_summary_barth(self, timeline_pd, tfile_prefix):
        filename = tfile_prefix + '_tf_op_duration_bar.png'
        title_ = tfile_prefix + 'TF : op duration bar chart'
        ax = self.summarize_barh(timeline_pd, 'arg_op', title=title_, topk=50, logx=True, figsize=(10,
                                                                                                   10))
        ax.figure.savefig(filename, bbox_inches='tight')

    def plot_summary_pie(self, timeline_pd, tfile_prefix):
        filename = tfile_prefix + '_tf_op_duration_pie.png'
        title_ = tfile_prefix + 'TF : op duration pie chart'
        timeline_pd_known = timeline_pd[(~timeline_pd['arg_op'].str.contains('unknown'))]
        ax = self.summarize_pie(timeline_pd_known, 'arg_op', title=title_, topk=50, logx=True, figsize=(10,
                                                                                                        10))
        ax.figure.savefig(filename, bbox_inches='tight')

    def merge_two_csv_files(self, merged_filepath, a, b):
        merged = a.merge(b, on='op')
        merged['speedup'] = merged['elapsed_time_stock'] / merged['elapsed_time_mkl']
        if merged['mkl_op_x'] is True:
            merged['mkl_op'] = True
        merged['mkl_op'] = merged['mkl_op_x'] + merged['mkl_op_y']
        merged = merged.drop(columns=['speedup_x', 'speedup_y'])
        merged = merged.drop(columns=['mkl_op_x', 'mkl_op_y'])
        if self.showAbsNumber is False:
            ret = merged.drop(columns=['elapsed_time_stock', 'elapsed_time_mkl'])
        else:
            ret = merged
        merged.to_csv(merged_filepath, index=False)
        return ret

    def create_csv_among_extra_common_ops(self, extra, common, fpath, tag1, tag2):
        import pandas as pd
        extra = extra.rename(columns={extra.columns.values[1]: tag1})
        extra = extra.drop(columns=['speedup'])
        common_time = common[common.columns.values[1]].sum(axis=0)
        append_op = 'Common ops with '+tag2
        to_append = [append_op, common_time, True]
        series = pd.Series(to_append, index = extra.columns)
        extra = extra.append(series, ignore_index=True)
        extra.to_csv(fpath, index=False)
        return extra

    def merge_two_csv_files_v2(self, merged_filepaths, a, b, tags=['stock','intel']):
        merged = a.merge(b, on='op')
        extra_a = a[~a.op.isin(merged.op)]
        common_a = a[a.op.isin(merged.op)]
        extra_b = b[~b.op.isin(merged.op)]
        common_b = b[b.op.isin(merged.op)]
        merged['speedup'] = merged.iloc[:,1] / merged.iloc[:,4]
        merged = merged.rename(columns={merged.columns.values[1]: tags[0], merged.columns.values[4]: tags[1]})
        if merged.iloc[:,3] is True:
            merged['mkl_op'] = True
        merged['mkl_op'] = merged.iloc[:,3] + merged.iloc[:,6]
        merged = merged.drop(columns=[merged.columns.values[2], merged.columns.values[3], merged.columns.values[5], merged.columns.values[6]])
        if self.showAbsNumber is False:
            ret = merged.drop(columns=[merged.columns.values[1], merged.columns.values[4]])
        else:
            ret = merged
        merged.to_csv(merged_filepaths[0], index=False)

        fpath = merged_filepaths[1]
        self.create_csv_among_extra_common_ops(extra_a, common_a, fpath, tags[0], tags[1])

        fpath = merged_filepaths[2]
        self.create_csv_among_extra_common_ops(extra_b, common_b, fpath, tags[1], tags[0])

        return ret

    def compare_bar_pie_charts(self, chart_type):
        import matplotlib.pyplot as plt
        import matplotlib.image as img
        from matplotlib import rcParams
        import os
        if chart_type == 'bar':
            imgfiles = [x for x in os.listdir('.') if '_tf_op_duration_bar.png' == x[-23:]]
        elif chart_type == 'pie':
            imgfiles = [x for x in os.listdir('.') if '_tf_op_duration_pie.png' == x[-23:]]
        else:
            return
        rcParams['figure.figsize'] = (30, 30)
        fig, ax = plt.subplots(1, 2)
        index = 0
        for imgf in imgfiles:
            image = img.imread(imgf)
            ax[index].imshow(image)
            index = index + 1

    def plot_compare_bar_charts(self, fpath, tags=['stock','intel'], num_hotspots=20):
        if self.showAbsNumber is False:
            return
        import numpy as np
        import matplotlib.pyplot as plt
        import csv
        reader = csv.DictReader(open(fpath))
        xlabels = []
        a_means = []
        b_means = []
        item_name = reader.fieldnames[0]
        a_name = reader.fieldnames[1]
        b_name = reader.fieldnames[2]
        index = 0
        for row in reader:
            if row['op'] != 'unknown' and index < num_hotspots:
                xlabels.append(row[item_name] + "_(mkl-" + str(row['mkl_op']) + ')')
                b_means.append(float(row[b_name]))
                a_means.append(float(row[a_name]))
                index = index + 1

        N = len(xlabels)
        ind = np.arange(N)
        width = 0.35
        fig = plt.figure(figsize=(18, 15))
        ax = fig.add_subplot(111)
        rects1 = ax.bar(ind, a_means, width, color='orange')
        rects2 = ax.bar(ind + width, b_means, width, color='royalblue')
        ax.set_ylabel('Elpased Time (ms)', fontsize=20)
        ax.set_title('TF Ops Performance Comparison ', fontdict={'fontsize': 28})
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(xlabels, rotation=45, rotation_mode="anchor")
        ax.legend((rects1[0], rects2[0]), [tags[0], tags[1]], fontsize=20)
        filename = 'compared_tf_op_duration_bar.png'
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
        plt.show()

    def plot_compare_ratio_bar_charts(self, fpath, tags=['stock','intel'], num_hotspots=20, max_speedup=100):
        import numpy as np
        import matplotlib.pyplot as plt
        import csv
        reader = csv.DictReader(open(fpath))
        xlabels = []
        b_xlabels = []
        c_xlabels = []
        a_means = []
        b_means = []
        c_means = []
        item_name = reader.fieldnames[0]
        c_name = reader.fieldnames[3]
        index = 0
        for row in reader:
            if row['op'] != 'unknown' and index < num_hotspots:

                if float(row[c_name]) > max_speedup:
                   speedup_val = max_speedup
                else:
                   speedup_val = float(row[c_name])

                if str(row['mkl_op']) == 'True':
                    #xlabels.append(row[item_name] + "_(mkl-" + str(row['mkl_op']) + ')')
                    b_xlabels.append(row[item_name])
                    b_means.append(speedup_val)
                    a_means.append(1)
                else:
                    c_xlabels.append(row[item_name])
                    c_means.append(speedup_val)
                    a_means.append(1)
                index = index + 1
        xlabels = b_xlabels + c_xlabels
        b_N = len(b_xlabels)
        c_N = len(c_xlabels)
        N = len(xlabels)
        b_ind = np.arange(b_N)
        ind = np.arange(N)
        c_ind = ind[b_N:]
        width = 0.35
        fig = plt.figure(figsize=(18, 15))
        ax = fig.add_subplot(111)
        rects2 = ax.bar(b_ind + width / 2, b_means, width, color='royalblue')
        rects3 = ax.bar(c_ind + width / 2 , c_means, width, color='orange')
        rects = rects2 + rects3
        ax.set_ylabel('Speedup',  fontsize=20)
        ax.set_title('TF Ops Performance Comparison ', fontdict={'fontsize': 28})
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(xlabels, rotation=45, rotation_mode="anchor")
        ax.legend([rects[0]],[tags[1]], fontsize=20)
        plt.axhline(y=1, linewidth=4, color='r')
        filename = 'compared_tf_op_duration_ratio_bar.png'
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
        plt.show()

    def plot_pie_chart(self, fpath, tag):
        import matplotlib.pyplot as plt
        import csv
        import matplotlib as mpl
        mpl.rcParams['font.size'] = 9.0
        reader = csv.DictReader(open(fpath))
        xlabels = []
        a_means = []
        item_name = reader.fieldnames[0]
        a_name = reader.fieldnames[1]

        for row in reader:
            if row['op'] != 'unknown':
                xlabels.append(row[item_name])
                a_means.append(float(row[a_name]))

        fig = plt.figure(figsize=(18, 15))

        ax1 = fig.add_axes([0, 0, .5, .5], aspect=1)
        wedges, texts, autotexts = ax1.pie(
            a_means, autopct='%1.1f%%',
            textprops=dict(color="w",  fontsize=18), radius=1.2)
        ax1.set_title(tag, fontdict={'fontsize': 28})

        box = ax1.legend(
            wedges,
            xlabels,
            title="TF Ops",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize=20)
        box.get_title().set_fontsize(20)
        filename = 'tf_op_duration_pie.png'
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
        plt.show()

    def plot_compare_pie_charts(self, fpath, tags=['stock','intel']):
        import matplotlib.pyplot as plt
        import csv
        import matplotlib as mpl
        mpl.rcParams['font.size'] = 9.0
        reader = csv.DictReader(open(fpath))
        xlabels = []
        a_means = []
        b_means = []
        item_name = reader.fieldnames[0]
        a_name = reader.fieldnames[1]
        b_name = reader.fieldnames[2]

        for row in reader:
            if row['op'] != 'unknown':
                xlabels.append(row[item_name])
                b_means.append(float(row[b_name]))
                a_means.append(float(row[a_name]))

        fig = plt.figure(figsize=(18, 15))

        ax1 = fig.add_axes([0, 0, .5, .5], aspect=1)
        wedges, texts, autotexts = ax1.pie(
            a_means, autopct='%1.1f%%',
            textprops=dict(color="w",  fontsize=18), radius=1.2)
        ax1.set_title(tags[0], fontdict={'fontsize': 28})

        ax2 = fig.add_axes([.5, .0, .5, .5], aspect=1)
        wedges, texts, autotexts = ax2.pie(
            b_means, autopct='%1.1f%%',
            textprops=dict(color="w", fontsize=18), radius=1.2)
        ax2.set_title(tags[1], fontdict={'fontsize': 28})

        box = ax2.legend(
            wedges,
            xlabels,
            title="TF Ops",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize=20)
        box.get_title().set_fontsize(20)
        filename = 'compared_tf_op_duration_pie.png'
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
        plt.show()


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
        import pandas as pd
        import numpy as np
        fig = plt.figure(figsize=(18, 15))
        ax = fig.add_subplot(111)
        figsize=(10,10)
        topk=50
        if Type == "time":
            print()
            print(' breakdown:',Group)
            if data['time'].dtypes != np.dtype('float64'):
                 data['time'] = data['time'].astype(float)
            time = data.groupby(Group)['time'].sum().sort_values().head(topk)
            print(time)
            title=Group + "Time Breakdown"
            time[:topk].plot.pie(
                ax=ax, title=title, figsize=figsize, logx=True, textprops=dict(fontsize=18),autopct='%1.1f%%')
            ax.figure.savefig(title)
        elif Type == "count":
            print()
            count = data[Group].value_counts().head(topk)
            print(count)
            title=Group+"Count Breakdown"
            count[:topk].plot.bar(
                ax=ax, title=title, figsize=figsize, logx=False, rot=45)
            ax.figure.savefig(title)
        return

    def stats_comp(self, name, Type,onednn_log1, onednn_log2, n=50, tags=['run1', 'run2']):
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        fig = plt.figure(figsize=(18, 15))
        ax = fig.add_subplot(111)
        figsize=(10,10)
        topk=50
        column_name = tags[1]+'/'+tags[0]
        
        d1 = onednn_log1.exec_data
        log1 = onednn_log1.filename
        d2 = onednn_log2.exec_data
        log2 = onednn_log2.filename
        print(name, 'stats:')
        if Type == "count":
            jitstat = pd.concat((d1[name].value_counts(), d2[name].value_counts()), axis=1, sort=True)
            jitstat.columns = ('1-' + log1, '2-' + log2)
            jitstat[column_name] = jitstat.iloc[:, 1] / jitstat.iloc[:, 0]
            jitstat_count = jitstat.sort_values('1-' + log1, ascending=False).head(n)
            #print(jitstat_count)
        elif Type == "time":
            if d1['time'].dtypes != np.dtype('float64'):
                 d1['time'] = d1['time'].astype(float)
            if d2['time'].dtypes != np.dtype('float64'):
                 d2['time'] = d2['time'].astype(float)
            jitstat = pd.concat((d1.groupby(name)['time'].sum(), d2.groupby(name)['time'].sum()), axis=1, sort=True)
            jitstat.columns = ('1-' + log1, '2-' + log2)
            jitstat[column_name ] = jitstat.iloc[:, 1] / jitstat.iloc[:, 0]
            jitstat_time = jitstat.sort_values('1-' + log1, ascending=False).head(n)
            #print(jitstat_time)
            title=name + column_name +" Time Comparison"
            jitstat_compare = jitstat_time.drop(columns=['1-' + log1, '2-' + log2])
            jitstat_compare[:topk].plot.bar(
                ax=ax, title=title, figsize=figsize, logx=False, rot=45)
            filename = name + " Time Comparison"
            ax.figure.savefig(filename)
        return

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
