#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import inspect


def has_nondefault_params(sgn):
    for v in sgn.parameters.values():
        if v.default is inspect._empty:
            return True
    return False


def run_examples(example_description, glbls_dict):
    parser = argparse.ArgumentParser(
        description=example_description,
    )
    parser.add_argument(
        "-r",
        "--run",
        type=str,
        help="Functions to execute. Use --run all to run all of them.",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List available function names to run",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Do not echo example name."
    )
    args = parser.parse_args()

    if args.list or not args.run:
        fns = []
        for n in glbls_dict:
            if inspect.isfunction(glbls_dict.get(n)):
                fns.append(n)
        if fns:
            print("Available examples:")
            print(", ".join(fns))
        else:
            print("No examples are availble.")
        exit(0)
    if args.run == "all":
        fns = []
        for n in glbls_dict:
            if inspect.isfunction(glbls_dict.get(n)):
                fns.append(n)
        args.run = fns
    else:
        args.run = args.run.split()

    if args.run:
        for fn in args.run:
            if fn in glbls_dict:
                clbl = glbls_dict.get(fn)
                sgn = inspect.signature(clbl)
                print("")
                if has_nondefault_params(sgn):
                    if not args.quiet:
                        print(
                            f"INFO: Skip exectution of {fn} as it "
                            "requires arguments"
                        )
                else:
                    if not args.quiet:
                        print(f"INFO: Executing example {fn}")
                    clbl()
                    if not args.quiet:
                        print("INFO: ===========================")

    else:
        raise ValueError("No function to run was specified")