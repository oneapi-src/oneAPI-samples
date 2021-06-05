#! /usr/bin/env python
import os, sys
import argparse


class TF_PB_Utils:

    def __init__(self):
        self.graph_def = None
        self.graph = None
        self.fnames = ['op_type', 'op_name', 'op1', 'op2',  'op3',  'op4',  'op5',  'op6']
        return

    def parse_pb_in_savemodel(self, folderpath):
        print('load SaveModel from : {}'.format(folderpath))
        import tensorflow as tf
        graph = None
        graph = tf.compat.v1.Graph()
        with tf.compat.v1.Session(graph=graph) as sess:
            tf.compat.v1.saved_model.loader.load(sess, [tf.compat.v1.saved_model.tag_constants.SERVING], folderpath)
        return graph

    def parse_pb_in_graphdef(self, filepath):
        print('Loading graph definition from :{}'.format(filepath))
        import tensorflow as tf
        graph_def = None
        graph = None
        try:
            with tf.compat.v1.gfile.GFile(filepath, "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())

            if graph_def == None:
                print(" failures at init a graph_def")
            print('Importing graph')
            with tf.Graph().as_default() as graph:
                print(' before import_graph_def ')
                tf.import_graph_def(
                    graph_def,
                    input_map=None,
                    return_elements=None,
                    name='',
                    op_dict=None,
                    producer_op_list=None
                )
                print(' after import_graph_def ')
            print("finish Importing ")
        except BaseException as e:
            print(' Error loading the graph definition: {}'.format(str(e)))
            print(" Errors for GraphDef Parsing")
            pass
        return graph

    def parse_pb(self, filepath):
        print('Parsing PB file : {}'.format(filepath))
        graph = None
        graph = self.parse_pb_in_graphdef(filepath)
        if graph == None:
            print("Need to parse it as save model")
            filename = filepath.split(os.sep)[-1]

            folderpath = filepath[:(len(filepath) - len(filename))]
            print(folderpath)
            if filename != 'saved_model.pb':
                print('need to make symbolic link')
                dst = folderpath + os.sep + 'saved_model.pb'
                if os.path.islink(dst):
                    print("link exists. unlink it : ", dst)
                    os.unlink(dst)
                os.symlink(filename, dst)
            graph = self.parse_pb_in_savemodel(folderpath)
        return graph
        
    def dump_ops_in_graph_into_csv(self, graph, filename='out.csv'):
        import tensorflow as tf
        import csv
        ops = graph.get_operations()  # type: Iterable[tf.Operation]
        f = open(filename, 'w')
        with f:
            writer = csv.DictWriter(f, fieldnames=self.fnames)
            writer.writeheader()
            for op in ops:
                #print('- {0} {1} ({2} outputs)'.format(op.type, op.name, len(op.outputs)))
                op_name = op.name.split('/')[-1]
                count = op.name.count('/')
                if count > 6:
                    count = 6
                op_list = [0 for i in range(6)]
                for i in range(count):
                    op_list[i] = op.name.split('/')[i]
                writer.writerow({'op_type': op.type, 'op_name': op_name, 'op1': op_list[0], 'op2': op_list[1], 'op3': op_list[2], 'op4': op_list[3], 'op5': op_list[4], 'op6': op_list[5]})

    def parse_ops_from_csv(self, filename='out.csv'):
        import pandas as pd
        data = pd.read_csv(filename, names=self.fnames)
        return data

    def draw_pie_chart(self, topo_data, group, topk=10):
        import matplotlib.pyplot as plt
        import numpy as np
        fig = plt.figure(figsize=(18, 15))
        ax = fig.add_subplot(111)
        figsize = (13, 15)
        title = group + "Count Breakdown"
        print(" draw pie chart : ", title)
        if topk > len(topo_data):
            topk = len(topo_data)
        topo_data[:topk].plot.pie(
            ax=ax, title=title, figsize=figsize, logx=True, textprops=dict(fontsize=18), autopct='%1.1f%%', labeldistance=1.1)
        box = ax.legend(
            topo_data[:topk].index,
            title="TF Ops",
            loc="best",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize=20)
        box.get_title().set_fontsize(20)
        plt.tight_layout()
        ax.figure.savefig(title)
        return

    def breakdown(self, data, columnindex, group, topk=5, keyword='', rowindex=-1):
        print("\n === Breakdown on Column: {0} , {1} === : ".format(columnindex, group))
        topo_data_list = []
        group_list = []
        groupdata = data.groupby(group).size().sort_values(ascending=False)
        #print(groupdata)
        group_size = groupdata.index.size
        print("group_size : ",group_size)
        if group_size > topk:
            group_size = topk
        if rowindex != -1:
            print('\n Group ops on row: {0} , {1} '.format(rowindex, groupdata.index[rowindex]))
            topo_data = data.loc[data[group] == groupdata.index[rowindex]]
            sorted_topo_data = topo_data.groupby('op_type').size().sort_values(ascending=False)
            print(sorted_topo_data)
            topo_data_list.append(sorted_topo_data)
            group_list.append(groupdata.index[rowindex])
        else:
            for i in range(group_size):
                #print('\n Group ops on: ',groupdata.index[i])
                print('\n Group ops on row: {0} , {1} '.format(i, groupdata.index[i]))
                topo_data = data.loc[data[group] == groupdata.index[i]]
                sorted_topo_data = topo_data.groupby('op_type').size().sort_values(ascending=False)
                print(sorted_topo_data)
                topo_data_list.append(sorted_topo_data)
                group_list.append(groupdata.index[i])
        return topo_data_list, group_list

    def get_data(self, filepath):
        data = None
        self.graph = self.parse_pb(filepath)
        if self.graph != None:
            self.dump_ops_in_graph_into_csv(self.graph)
            data = self.parse_ops_from_csv()
        return data

    def dump_columns(self, data, topk=10):
        for i in range(5):
            print("\n == Dump column : {0} == ".format(i))
            group = self.fnames[i]
            groupdata = data.groupby(group).size().sort_values(ascending=False).head(topk)
            print(groupdata)
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='The file name of the frozen graph/ The folder name contained the saved model')
    # will dump all columns if no column index is assigned, and group by second column (index=1)
    parser.add_argument("-c", "--column", type=int, choices=[0, 1, 2, 3, 4, 5], default=-1, help="Input column index")
    # will dump all rows if no row index is assigned
    parser.add_argument("-r", "--row", type=int, choices=[0, 1, 2, 3, 4, 5], default=-1, help="Input sored row index")
    args = parser.parse_args()

    tfpb = TF_PB_Utils()
    data = tfpb.get_data(args.file)
    if args.column == -1:
        tfpb.dump_columns(data)
        sys.exit()

    topo_data_list, group_list = tfpb.breakdown(data, args.column, tfpb.fnames[args.column], rowindex=args.row)
    if args.row == -1:
        sys.exit()

    for i in range(len(topo_data_list)):
        tfpb.draw_pie_chart(topo_data_list[i], group_list[i])

