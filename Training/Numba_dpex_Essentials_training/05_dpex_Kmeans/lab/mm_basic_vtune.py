import os
import ipywidgets as widgets
from IPython.display import IFrame

out0 = widgets.Output()
out1 = widgets.Output()
out2 = widgets.Output()
out3 = widgets.Output()
out4 = widgets.Output()
out5 = widgets.Output()


tab = widgets.Tab(children = [out0, out1, out2,out3, out4, out5])
tab.set_title(0, 'GPU offload')
tab.set_title(1, 'GPU hotspots')
tab.set_title(2, 'GPU repeat =20')
tab.set_title(3, 'GPU hotspots repeat= 20')
tab.set_title(4, 'GPU repeat =100')
tab.set_title(5, 'GPU hotspots repeat= 100')

display(tab)

with out0:
    display(IFrame(src='reports/kmeans_repeat1_offload.html', width=1024, height=768))
with out1:
    display(IFrame(src='reports/kmeans_repeat1_hotspots.html', width=1024, height=768))
with out2:
    display(IFrame(src='reports/kmeans_repeat20_offload.html', width=1024, height=768))
with out3:
    display(IFrame(src='reports/kmeans_repeat20_hotspots.html', width=1024, height=768))
with out4:
    display(IFrame(src='reports/kmeans_repeat100_offload.html', width=1024, height=768))
with out5:
    display(IFrame(src='reports/kmeans_repeat100_hotspots.html', width=1024, height=768))