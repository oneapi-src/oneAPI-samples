import os
import ipywidgets as widgets
from IPython.display import IFrame

out0 = widgets.Output()
out1 = widgets.Output()
out2 = widgets.Output()
out3 = widgets.Output()

tab = widgets.Tab(children = [out0, out1, out2,out3])
tab.set_title(0, 'GPU offload 2**14')
tab.set_title(1, 'GPU hotspots 2**14')
tab.set_title(2, 'GPU offload 2**24')
tab.set_title(3, 'GPU hotspots 2**24')

display(tab)

with out0:
    display(IFrame(src='reports/Knn_output_214.html', width=1024, height=768))
with out1:
    display(IFrame(src='reports/Knn_output_hotspots_214.html', width=1024, height=768))
with out2:
    display(IFrame(src='reports/Knn_output_224.html', width=1024, height=768))
with out3:
    display(IFrame(src='reports/Knn_output_hotspots_224.html', width=1024, height=768))
