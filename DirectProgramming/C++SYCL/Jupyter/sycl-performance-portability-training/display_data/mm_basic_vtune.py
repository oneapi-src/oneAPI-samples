import os
import ipywidgets as widgets
from IPython.display import IFrame

out0 = widgets.Output()
out1 = widgets.Output()
out2 = widgets.Output()
out3 = widgets.Output()
out4 = widgets.Output()
out5 = widgets.Output()

tab = widgets.Tab(children = [out0, out1, out2, out3, out4, out5])
tab.set_title(0, 'Gen9 1024x1024')
tab.set_title(1, 'Gen9 5120x5120')
tab.set_title(2, 'Gen9 10240x10240')
tab.set_title(3, 'DG1 1024x1024')
tab.set_title(4, 'DG1 5120x5120')
tab.set_title(5, 'DG1 10240x10240')

display(tab)

with out0:
    display(IFrame(src='reports/vtune_mm_dpcpp_basic_Gen9_1024.html', width=1024, height=768))
with out1:
    display(IFrame(src='reports/vtune_mm_dpcpp_basic_Gen9_5120.html', width=1024, height=768))
with out2:
    display(IFrame(src='reports/vtune_mm_dpcpp_basic_Gen9_10240.html', width=1024, height=768))
with out3:
    display(IFrame(src='reports/vtune_mm_dpcpp_basic_DG1_1024.html', width=1024, height=768))
with out4:
    display(IFrame(src='reports/vtune_mm_dpcpp_basic_DG1_5120.html', width=1024, height=768))
with out5:
    display(IFrame(src='reports/vtune_mm_dpcpp_basic_DG1_10240.html', width=1024, height=768))