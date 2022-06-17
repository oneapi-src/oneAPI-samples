import os
import ipywidgets as widgets
from IPython.display import IFrame

out0 = widgets.Output()

tab = widgets.Tab(children = [out0])
tab.set_title(0, 'GPU')


display(tab)

with out0:
    display(IFrame(src='reports/roofline_pairwise.html', width=1024, height=768))

