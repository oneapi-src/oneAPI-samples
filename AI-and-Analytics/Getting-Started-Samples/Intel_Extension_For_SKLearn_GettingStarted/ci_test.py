def runJupyterNotebook(input_notebook_filename, output_notebook_filename, fdpath='./'):
    import nbformat
    import os
    from nbconvert.preprocessors import ExecutePreprocessor
    from nbconvert.preprocessors import CellExecutionError
    if os.path.isfile(input_notebook_filename) is False:
        print("No Jupyter notebook found : ",input_notebook_filename)
    try:
        with open(input_notebook_filename) as f:
            nb = nbformat.read(f, as_version=4)
            ep = ExecutePreprocessor(timeout=6000, allow_errors=True)
            ep.preprocess(nb, {'metadata': {'path': fdpath}})
            with open(output_notebook_filename, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
            return 0
    except CellExecutionError:
        print("Exception!")
        return -1


runJupyterNotebook('Intel_Extension_For_SKLearn_GettingStarted.ipynb', 'Intel_Extension_For_SKLearn_GettingStarted_result.ipynb' )