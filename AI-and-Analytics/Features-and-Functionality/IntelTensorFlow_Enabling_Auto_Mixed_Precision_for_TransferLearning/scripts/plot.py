import pandas as pd

throughput_list = []
log_file = open('log.txt', 'r')
lines = log_file.readlines()
for line in lines:
    if 'Throughput' in line:
        throughput = line.split(': ')[1]
        throughput_list.append(float(throughput))

if len(throughput_list) == 2:
    speedup = float(throughput_list[1])/float(throughput_list[0])
    print("Speedup : ", speedup)
    df = pd.DataFrame({'pretrained_model':['saved model', 'optimized model'], 'Speedup':[1, speedup]})
    ax = df.plot.bar( x='pretrained_model', y='Speedup', rot=0)
else:
    print("Incorrect data size to calculate speedup")