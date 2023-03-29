# !/usr/bin/env python3
import os
import json
from collections import OrderedDict
from pathlib import Path
import pandas as pd
import pathlib
import sys
import pprint
from datetime import datetime
# pp = pprint.PrettyPrinter(indent=4)
# pp.pprint

def make_json_list(basedir:str):
    ''' Walk basedir and create list of all filepaths to JSON code samples.'''
    json_file_paths = []
    try:
        if os.path.isdir(basedir):
            for root, dirs, files in os.walk(basedir):
                for file in files:
                    if file.endswith("sample.json"):
                        filepath = os.path.join(root,file)
                        json_file_paths.append(filepath) 
            filtered_list = [j for j in json_file_paths if not j.startswith("./oneAPI-samples/Publications")]
            print(f"\n Test {filtered_list}")
            return filtered_list
    except Exception as e:
        print(f"Error. Ensure root directory is oneAPI-samples repo. \n: {e}")

def merge_json_files(filepaths:list):
    '''Create pre-prod database from merged sample.json files from list of filenames'''
    results_list = []
    try:
        for f in filepaths:
            print(f)
            with open(f, 'r') as infile:
                results_list.append(json.load(infile))
                # print(results_list)
        with open('sample_db_pre.json', 'w') as output_file:
            json.dump(results_list, output_file)
        return
    except Exception as e:
        print(f"An error occurred. Ensure make_json_list() executes successfully. \n: {e}")

def make_url_dict(branch:str,file_paths:list):
    '''Build dict where key is 'name' from sample.json and value is its published url on Github'''
    baseurl = "https://github.com/oneapi-src/oneAPI-samples/tree/{}".format(branch)     
    list_name = []
    list_urls = [] 
    if file_paths is not None:
        for f in file_paths:
            path_base = pathlib.PurePath(f)
            # TODO: Revise filepath path SLICE => [1:-1] after migration => oneAPI-samples
            path_slice = '/'.join(path_base.parts[1:-1])
            full_url = os.path.join(baseurl,path_slice)
            list_urls.append(str(full_url))
            with open(f) as file:
                contents = json.load(file)
                tlkt_name = contents["name"]
                list_name.append(tlkt_name)
        raw_dict = dict(zip(list_name,list_urls))
        url_dict = OrderedDict(sorted(raw_dict.items()))
        return url_dict
    else:
        print("An error occurred. Ensure file_paths are generated.")

def df_sort_filter():
    '''Import JSON to DF; sort by name col; filter only records w/ expertise; drop unused columns; add url col.'''
    raw_data = pd.read_json("sample_db_pre.json")
    df = pd.DataFrame(raw_data)
    df = df.sort_values(by=['name'], ignore_index=True,key=lambda x: x.str.lower())
    df = df.dropna(subset=['expertise']) # DROP row if 'expertise' shows "NaN"
    df = df.drop(["guid","toolchain", "os", "builder", "ciTests","commonFolder", "dependencies", "categories"], axis=1)
    df['url'] = df.insert(2, 'url', 'np.Nan')
    return df

def df_add_urls(file_paths:list):
    '''Load all names, urls in make_url_dict(); prep df_sort_filter() and apply urls per their names to new df'''
    new_dict = make_url_dict("master", file_paths)
    new_df = df_sort_filter()
    new_df['url'] = new_df['name'].apply(lambda x:new_dict.get(x)) 
    return new_df 

def df_to_db(file_paths:list):
    '''Create prod database, combining df_add_urls(); output for frontend display'''
    df = df_add_urls(file_paths)
    rev_json = Path('src/docs/_static/sample_db_prd.json')
    db = df.to_json(rev_json, orient='records')    
    return db

def main():
    '''Orchestrate sequence of steps to output sample_db_prd.json'''
    rootdir = sys.argv[-1]
    file_paths = make_json_list(rootdir)
    print(file_paths)
    # print("OBJ:",type(file_paths))
    merge_json_files(file_paths)
    # print("Check complete...?")
    json_db = df_to_db(file_paths)

    return json_db

if __name__ == "__main__":
    main()
