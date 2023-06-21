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


import requests
import pandas as pd




def generate_html(report_folder, images):
    import pandas as pd
    from IPython.core.display import display,HTML


    df = pd.DataFrame()
    #df['ff'] = table_df
    df['Diagram'] = images
    # convert your links to html tags
    def path_to_image_html(path):
        return '<img src="'+ path + '" width="1000" >'

    pd.set_option('display.max_colwidth', None)

    image_cols = ['Diagram']  #<- define which columns will be used to convert to html

    # Create the dictionariy to be passed as formatters
    format_dict = {}
    for image_col in image_cols:
        format_dict[image_col] = path_to_image_html


    df.to_html(report_folder + os.sep + 'report.html', escape=False, formatters=format_dict)
    return

def uncompress_gz(fname, fname_prefix=''):
    import gzip
    f_name = fname.replace(".gz","")
    fn_nofd = fname_prefix + f_name.split(os.sep)[-1]
    g_file = gzip.GzipFile(fname)
    open(fn_nofd,"wb+").write(g_file.read())
    g_file.close()
    return fn_nofd

class TFTimelinePresenter:

    def __init__(self, showAbsNumber=False, parsed_pd_col='arg_op', parsed_pd_col2=''):
        self.showAbsNumber = showAbsNumber
        self.parsed_pd_col = parsed_pd_col
        self.parsed_pd_col2 = parsed_pd_col2
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

    def summarize_items(self, tl, items, ascending=False):
        items = tl.groupby(items)['dur'].sum().sort_values(ascending=ascending)
        return items


    def summarize_item(self, tl, item, ascending=False):
        items = tl.groupby([item])['dur'].sum().sort_values(ascending=ascending)
        #df_items = items.reset_index(name='time')
        return items #, df_items

    def summarize_barh(self, tl, item, topk=15, ascending=False, ax=None, title=None, figsize=None, logx=False):
        ret = self.summarize_item(tl, item, ascending)[:topk].plot.barh(
             title=title, figsize=figsize, logx=False)
        return ret

    def summarize_pie(self, tl, item, topk=15, ascending=False, ax=None, title=None, figsize=None, logx=False):
        ret = self.summarize_item(tl, item, ascending)[:topk].plot.pie(
             title=title, figsize=figsize, logx=logx, autopct='%1.1f%%')
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
        has_long_name = True
        try:
            t['arg_long_name'] = t['arg_long_name'].astype('str').str.split(pat=':').str.get(-1)
        except:
            has_long_name = False
            pass

        if 'arg_shape' in t.columns:
            has_shape = True
        else:
            has_shape = False

        t['arg_opname'] = t['arg_name'].apply(self.opname)
        t['arg_opbase'] = t['arg_name'].apply(self.opbase)
        return t, has_long_name, has_shape

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
        #print(timeline_pd)

        if self.parsed_pd_col2 != '':
            sitems_col2 = self.summarize_items(timeline_pd, [self.parsed_pd_col, self.parsed_pd_col2])
        else:
            sitems_col2 = None
        sitems = self.summarize_item(timeline_pd, self.parsed_pd_col)
        #print(sitems)
        import csv
        filename = fn.split('.')[0] + '.csv'
        f = open(filename, 'w')
        time_col_name = 'elapsed_time_' + tfile_prefix
        total_time = 0.0
        total_mkl_time = 0.0
        with f:
            fnames = ['op', time_col_name, 'speedup', 'mkl_op', 'native_op']
            writer = csv.DictWriter(f, fieldnames=fnames)
            writer.writeheader()
            x = 0
            for sitem in sitems:
                mkl_op = False
                native_op = False
                #if tfile_prefix == 'mkl':
                op_name = sitems.index[x].strip('_')
                if op_name.find('Mkl') != -1:
                    mkl_op = True
                    op_name = op_name[3:]
                    total_mkl_time += float(sitems[x])
                    if op_name.find('Native') != -1:
                        op_name = op_name[6:]
                        native_op = True
                total_time += float(sitems[x])
                #else:
                #    op_name = sitems.index[x].strip('_')
                writer.writerow({'op': op_name, time_col_name: sitems[x], 'speedup': 0, 'mkl_op': mkl_op, 'native_op': native_op})
                x = x + 1

        percentage_filename = ''
        if total_mkl_time > 0.0:
            percentage_filename = fn.split('.')[0] +'_mkl_percentage' +'.csv'
            f = open(percentage_filename, 'w')
            with f:
                fnames = ['op', 'time']
                writer = csv.DictWriter(f, fieldnames=fnames)
                writer.writeheader()
                writer.writerow({'op': 'mkl_op', 'time': total_mkl_time})
                writer.writerow({'op': 'nono_mkl_op', 'time': total_time - total_mkl_time})

        ret = None, percentage_filename, None
        if self.showAbsNumber is True:
            ret = sitems, percentage_filename, sitems_col2
        return ret

    def plot_summary_barth(self, timeline_pd, tfile_prefix):
        filename = tfile_prefix + '_tf_op_duration_bar.png'
        title_ = tfile_prefix + 'TF : op duration bar chart'
        ax = self.summarize_barh(timeline_pd, self.parsed_pd_col, title=title_, topk=50, logx=True, figsize=(10,
                                                                                                   10))
        ax.figure.savefig(filename, bbox_inches='tight')

    def plot_summary_pie(self, timeline_pd, tfile_prefix):
        filename = tfile_prefix + '_tf_op_duration_pie.png'
        title_ = tfile_prefix + 'TF : op duration pie chart'
        timeline_pd_known = timeline_pd[(~timeline_pd[self.parsed_pd_col].str.contains('unknown'))]
        ax = self.summarize_pie(timeline_pd_known, self.parsed_pd_col, title=title_, topk=50, logx=True, figsize=(10,
                                                                                                        10))
        ax.figure.savefig(filename, bbox_inches='tight')

    def get_diff_from_csv_filenames(self, x, y):
        x_split = x.split('_')
        y_split = y.split('_')
        if len(x_split) != len(y_split):
            print("ERROR! can't two files have different formats")
            return '', ''
        for i in range(len(x_split)):
            if x_split[i] != y_split[i]:
                break
        return x_split[i], y_split[i]

    def merge_two_csv_files(self, merged_filepath, a, b):
        merged = a.merge(b, on='op')
        merged['speedup'] = merged['elapsed_time_stock'] / merged['elapsed_time_mkl']
        if merged['mkl_op_x'] is True:
            merged['mkl_op'] = True
        merged['mkl_op'] = merged['mkl_op_x'] + merged['mkl_op_y']
        if merged['native_op_x'] is True:
            merged['native_op'] = True
        merged['native_op'] = merged['native_op_x'] + merged['native_op_y']
        merged = merged.drop(columns=['speedup_x', 'speedup_y'])
        merged = merged.drop(columns=['mkl_op_x', 'mkl_op_y'])
        merged = merged.drop(columns=['native_op_x', 'native_op_y'])
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
        append_op = 'Common ops with ' + tag2
        to_append = [append_op, common_time, True, False]
        series = pd.Series(to_append, index=extra.columns)
        extra = extra.append(series, ignore_index=True)
        extra.to_csv(fpath, index=False)
        return extra

    def merge_two_csv_files_v2(self, merged_filepaths, a, b, tags=['stock', 'intel']):
        merged = a.merge(b, on='op')
        extra_a = a[~a.op.isin(merged.op)]
        common_a = a[a.op.isin(merged.op)]
        extra_b = b[~b.op.isin(merged.op)]
        common_b = b[b.op.isin(merged.op)]
        merged['speedup'] = merged.iloc[:, 1] / merged.iloc[:, 5]
        merged = merged.rename(columns={merged.columns.values[1]: tags[0], merged.columns.values[5]: tags[1]})
        if merged.iloc[:, 3] is True:
            merged['mkl_op'] = True
        merged['mkl_op'] = merged.iloc[:, 3] + merged.iloc[:, 7]
        if merged.iloc[:, 4] is True:
            merged['native_op'] = True
        merged['native_op'] = merged.iloc[:, 4] + merged.iloc[:, 8]
        merged = merged.drop(columns=[merged.columns.values[2], merged.columns.values[3], merged.columns.values[4], merged.columns.values[6], merged.columns.values[7], merged.columns.values[8]])
        if self.showAbsNumber is False:
            ret = merged.drop(columns=[merged.columns.values[1], merged.columns.values[5]])
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

    def plot_compare_bar_charts(self, fpath, tags=['stock', 'intel'], num_hotspots=20, report_folder="report"):
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
        plt.savefig(report_folder + os.sep + filename, bbox_inches='tight', pad_inches=0.1)
        #plt.show()
        return filename

    def plot_compare_ratio_bar_charts(self, fpath, tags=['stock', 'intel'], num_hotspots=20, max_speedup=100, report_folder="report"):
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
        N = len(xlabels)
        b_ind = np.arange(b_N)
        ind = np.arange(N)
        c_ind = ind[b_N:]
        width = 0.35
        fig = plt.figure(figsize=(18, 15))
        ax = fig.add_subplot(111)
        rects2 = ax.bar(b_ind + width / 2, b_means, width, color='royalblue')
        rects3 = ax.bar(c_ind + width / 2, c_means, width, color='orange')
        rects = rects2 + rects3
        ax.set_ylabel('Speedup', fontsize=20)
        ax.set_title('TF Ops Performance Comparison ', fontdict={'fontsize': 28})
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(xlabels, rotation=45, rotation_mode="anchor")
        ax.legend([rects[0]], [tags[1]], fontsize=20)
        plt.axhline(y=1, linewidth=4, color='r')
        filename = 'compared_tf_op_duration_ratio_bar.png'
        plt.savefig(report_folder + os.sep + filename, bbox_inches='tight', pad_inches=0.1)
        #plt.show()
        return filename

    def plot_pie_chart(self, fpath, tag, report_folder="report", target_col='op'):
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
            if row[target_col] != 'unknown':
                xlabels.append(row[item_name])
                a_means.append(float(row[a_name]))

        fig = plt.figure(figsize=(18, 15))

        ax1 = fig.add_axes([0, 0, .5, .5], aspect=1)
        wedges, texts, autotexts = ax1.pie(
            a_means, autopct='%1.1f%%',
            textprops=dict(color="w", fontsize=18), radius=1.2)
        ax1.set_title(tag, fontdict={'fontsize': 28})

        box = ax1.legend(
            wedges,
            xlabels,
            title="TF Ops",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize=20)
        box.get_title().set_fontsize(20)
        filename = tag + "_" + 'tf_op_duration_pie.png'
        plt.savefig(report_folder + os.sep + filename, bbox_inches='tight', pad_inches=0.1)
        #plt.show()
        return filename

    def plot_compare_pie_charts(self, fpath, tags=['stock', 'intel']):
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
            textprops=dict(color="w", fontsize=18), radius=1.2)
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
        #plt.show()

    def plot_dataframe_table(self, dataf, tag, report_folder="report"):
        import matplotlib.pyplot as plt
        import pandas as pd
        from pandas.plotting import table # EDIT: see deprecation warnings below

        ax = plt.subplot(111, frame_on=False) # no visible frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis

        table(ax, dataf)  # where df is your data frame

        filename = tag + "_" + 'dataframe.png'
        plt.savefig(report_folder + os.sep + filename, bbox_inches='tight', pad_inches=0.1, dpi=250)
        return filename


#if __name__ == '__main__':

def main():

    import os, sys
    import datetime
    images = []
    timeinfo = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    report_folder = "report_" + timeinfo
    os.makedirs(report_folder, exist_ok=True)

    parsed_pd_col = "arg_op"
    tfp = TFTimelinePresenter(True)
    if len(sys.argv) > 2 and '.json' in sys.argv[1] and '.json' in sys.argv[2]:
        print(sys.argv)

        import matplotlib.pyplot as plt
        csvfiles=[]
        sitems_list=[]
        sitems_col2_list=[]
        has_shape_list=[]
        fname_prefix = []
        selected_datafiles = []
        percentage_filename = ''

        #tfp = TFTimelinePresenter(True)
        fname_prefix.append("base_")
        fname_prefix.append("compare_")

        for index, sys_argv in enumerate(sys.argv):
            if index < 1:
                continue
            if '.gz' in sys_argv:
                fn_nofd = uncompress_gz(sys_argv, fname_prefix=fname_prefix[index - 1])
            else:
                import shutil
                fn_nofd = fname_prefix[index - 1] + sys_argv.split(os.sep)[-1]
                shutil.copyfile(sys_argv, fn_nofd)
            selected_datafiles.append(fn_nofd)


        for index, fn in enumerate(selected_datafiles):
            fn_nofd = fn.split(os.sep)[-1]
            tfile_name= fn_nofd.split('.')[0] #+ '_' + str(index)
            tfile_prefix = fn_nofd.split('_')[0]
            tfile_postfix = fn_nofd.strip(tfile_prefix)[1:]
            #csvpath = report_folder +os.sep+tfile_name+'.csv'
            csvpath = tfile_name+'.csv'
            print(csvpath)
            csvfiles.append(csvpath)
            timeline_pd, has_long_name, has_shape = tfp.postprocess_timeline(tfp.read_timeline(fn))
            if has_long_name is True:
                parsed_pd_col = "arg_long_name"
                tfp.parsed_pd_col = parsed_pd_col
            if has_shape is True:
                parsed_pd_col2 = "arg_shape"
                tfp.parsed_pd_col2 = parsed_pd_col2

            timeline_pd = timeline_pd[timeline_pd['ph'] == 'X']
            timeline_pd = timeline_pd[timeline_pd[parsed_pd_col] != 'nan']
            sitems, percentage_filename, sitems_col2 = tfp.get_tf_ops_time(timeline_pd,fn,tfile_prefix)
            sitems_list.append(sitems)
            sitems_col2_list.append(sitems_col2)
            has_shape_list.append(has_shape)

        if percentage_filename != '':
            print(percentage_filename)
            filename = tfp.plot_pie_chart(percentage_filename, 'mkl_percentage', report_folder=report_folder)
            images.append(filename)
        import pandas as pd

        csvarray = []
        csvfilenames= []
        for csvf in csvfiles:
            print("read into pandas :",csvf)
            a = pd.read_csv(csvf)
            csvarray.append(a)
            if csvf.find(os.sep) > 0:
                csvfilenames.append(csvf.split(os.sep)[-1])
            else:
                csvfilenames.append(csvf)

        a = csvarray[0]
        b = csvarray[1]
        tags=[]
        #from profile_utils import PerfPresenter
        #perfp=PerfPresenter()
        tag0, tag1 = tfp.get_diff_from_csv_filenames(csvfilenames[0][:-4],csvfilenames[1][:-4])
        tags = [tag0, tag1]
        print('tags : ',tags)
        import os
        import pandas as pd
        fdir='merged'
        if not os.path.exists(fdir):
            os.mkdir(fdir)

        fpaths=[]
        fpaths.append(fdir+os.sep+'merged.csv')
        fpaths.append(fdir+os.sep+'diff_'+tags[0]+'.csv')
        fpaths.append(fdir+os.sep+'diff_'+tags[1]+'.csv')
        #merged=tfp.merge_two_csv_files(fpath,a,b)
        merged=tfp.merge_two_csv_files_v2(fpaths, a, b, tags)

        print("Compare common operations between ", tags)
        merged_df = pd.read_csv(fpaths[0])
        #merged_df
        filename = tfp.plot_dataframe_table(merged_df, "common_ops", report_folder=report_folder)
        images.append(filename)

        print(fpaths[0])
        filename = tfp.plot_compare_bar_charts(fpaths[0], tags=tags, report_folder=report_folder)
        images.append(filename)
        filename = tfp.plot_compare_ratio_bar_charts(fpaths[0], tags=['','oneDNN ops'], max_speedup=20, report_folder=report_folder)
        images.append(filename)

        # First Timeline
        # elapsed time of TF ops from Stock TF or the first csv/Timline file
        filename = tfp.plot_pie_chart(csvfiles[0], tags[0], report_folder=report_folder)
        images.append(filename)

        # elapsed time of unique TF operations from Stock TF or the first csv/Timline file
        filename = tfp.plot_pie_chart(fpaths[1],"unique" + "_" + tags[0], report_folder=report_folder)
        images.append(filename)

        table_df = sitems_list[0].reset_index(name='time')
        filename = tfp.plot_dataframe_table(table_df, "tf_ops_1", report_folder=report_folder)
        images.append(filename)
       # clear plt
        plt.clf()

        #if sitems_col2_list[0] != None:
        if has_shape_list[0] is True:
            table_df2 = sitems_col2_list[0].reset_index(name='time')
            filename = tfp.plot_dataframe_table(table_df2, "tf_ops_shape_1", report_folder=report_folder)
            images.append(filename)
            # clear plt
            plt.clf()


        # Second Timeline
        # elapsed time of TF ops from Intel TF or the second csv/Timline file
        filename = tfp.plot_pie_chart(csvfiles[1], tags[1], report_folder=report_folder)
        images.append(filename)

        # elapsed time of unique TF operations from Intel TF or the seond csv/Timline file
        filename = tfp.plot_pie_chart(fpaths[2], "unique" + "_" + tags[1], report_folder=report_folder)
        images.append(filename)

        table_df = sitems_list[1].reset_index(name='time')
        filename = tfp.plot_dataframe_table(table_df, "tf_ops_2", report_folder=report_folder)
        images.append(filename)
       # clear plt
        plt.clf()

        #if sitems_col2_list[1] != None:
        if has_shape_list[1] is True:
            table_df2 = sitems_col2_list[1].reset_index(name='time')
            filename = tfp.plot_dataframe_table(table_df2, "tf_ops_shape_2", report_folder=report_folder)
            images.append(filename)
            # clear plt
            plt.clf()


    elif len(sys.argv) > 1 and '.json' in sys.argv[1]:
        print(sys.argv)
        import matplotlib.pyplot as plt
        #fn =  sys.argv[1]

        if '.gz' in sys.argv[1]:
            fname = uncompress_gz(sys.argv[1])
        else:
            import shutil
            fname = sys.argv[1].split(os.sep)[-1]
            shutil.copyfile(sys.argv[1], fname)

        tfile_prefix = fname.split('_')[0]
        tfile_postfix = fname.strip(tfile_prefix)[1:]
        #from profile_utils import TFTimelinePresenter
        #tfp = TFTimelinePresenter(True)
        #timeline_pd = tfp.postprocess_timeline(tfp.read_timeline(fn))
        timeline_pd, has_long_name, has_shape = tfp.postprocess_timeline(tfp.read_timeline(fname))
        if has_long_name is True:
            parsed_pd_col = "arg_long_name"
            tfp.parsed_pd_col = parsed_pd_col
        if has_shape is True:
            parsed_pd_col2 = "arg_shape"
            tfp.parsed_pd_col2 = parsed_pd_col2
        timeline_pd = timeline_pd[timeline_pd['ph'] == 'X']
        timeline_pd = timeline_pd[timeline_pd[parsed_pd_col] != 'nan']

        sitems, percentage_filename, sitems_col2 = tfp.get_tf_ops_time(timeline_pd,fname,tfile_prefix)


        filename= tfile_prefix +'_tf_op_duration_bar.png'
        title_=tfile_prefix +'TF : op duration bar chart'
        tfp.summarize_barh(timeline_pd, parsed_pd_col, title=title_, topk=50, logx=True, figsize=(10,10), ascending=True)
        print(filename)
        plt.savefig(report_folder + os.sep + filename,bbox_inches='tight')
        images.append(filename)
        # clear plt
        plt.clf()

        filename= tfile_prefix +'_tf_op_duration_pie.png'
        title_=tfile_prefix +'TF : op duration pie chart'
        timeline_pd_known = timeline_pd[ ~timeline_pd[parsed_pd_col].str.contains('unknown') ]
        tfp.summarize_pie(timeline_pd_known, parsed_pd_col, title=title_, topk=20, logx=True, figsize=(10,10))
        print(filename)
        plt.savefig(report_folder + os.sep + filename,bbox_inches='tight')
        images.append(filename)
        # clear plt
        plt.clf()

        table_df = sitems.reset_index(name='time')
        filename = tfp.plot_dataframe_table(table_df, "tf_ops", report_folder=report_folder)
        images.append(filename)
       # clear plt
        plt.clf()

        #if sitems_col2 != None:
        if has_shape is True:
            table_df2 = sitems_col2.reset_index(name='time')
            filename = tfp.plot_dataframe_table(table_df2, "tf_ops_shape", report_folder=report_folder)
            images.append(filename)
            # clear plt
            plt.clf()


    elif len(sys.argv) > 1:
        print(sys.argv)

    # generate html output file
    generate_html(report_folder, images)



if __name__ == '__main__':
    sys.exit(main())
