# Analyze TensorFlow Trace Json Files
This analyze tool helps users to analyze TensorFlow Trace Json with a HTML output which contains some statistic charts and a timeline chart.


## Prerequisites 

*  users need to enable TensorFlow Profiler in their workloads first.  Please refer to [TF_PerfAnalysis.ipynb](../TF_PerfAnalysis.ipynb) for more details.

## How to analyze TensorFlow tace json.gz files

### analyze a TensorFlow tace json.gz file
Users could also use a file path instead.  
*  parse a trace from workload : `$./analyze A.trace.json.gz`  


### Compare and Analyze two TensorFlow tace json.gz files 
Users could also use a file path instead. 
*  compare two json.gz files "A.trace.json.gz" and "B.trace.json.gz" : `$./analyze A.trace.json.gz B.trace.json.gz` 

## Understand Reports  

<details>
<summary> oneDNN overall useage  </summary>
  
Users could understand how many percentage this workload spend on oneDNN computations.   
Here is an example diagram, and more than 94% of cpu time are on oneDNN computations which is good.      
<br><img src="report_template/mkl_percentage_tf_op_duration_pie.png" width="400" height="300"><br>
</details>

<details>
<summary> Bart Chart for TF ops Elapsed time comparison </summary>
  
Users could compare TF ops elpased time between Base and Compare run.    
Here is an example diagram.  
Yellow bars are from Base Run and Blue bars are from Compare run.   
Overall, lower is better for the elapsed time.  
<br><img src="report_template/compared_tf_op_duration_bar.png" width="400" height="300"><br>
</details>

<details>
<summary> Bart Chart for TF ops speedup comparison  </summary>
  
Users could compare TF ops speedup between Base and Compare run.    
Here is an example diagram.  
Yellow bars are from Eigen Ops and Blue bars are oneDNN Ops.   
Each bar show the speedup from Compare run to Base run, so higher is better.       
<br><img src="report_template/compared_tf_op_duration_ratio_bar.png" width="400" height="300"><br>
</details>

<details>
<summary> Pie Chart for Base Run TF Ops hotspots  </summary>
  
Users could understand how many percentage this workload spend on different TF ops.   
Here is an example diagram, and more than 73% of cpu time are on FusedConv2D. 
Users could start optimize the top hotspot to improve the performance
<br><img src="report_template/base_tf_op_duration_pie.png" width="420" height="300"><br>
</details>

<details>
<summary>  Pie Chart for Base Run Unique TF Ops hotspots  </summary>
  
Users could understand how many percentage this workload spend on unique TF ops only used in the Base run.   
Here is an example diagram, and there is a unique TF ops "Add" token ~15% of total cpu time.    
<br><img src="report_template/unique_base_tf_op_duration_pie.png" width="480" height="300"><br>
</details>

<details>
<summary> Pie Chart for Compare Run TF Ops hotspots   </summary>
  
Users could understand how many percentage this workload spend on different TF ops.   
Here is an example diagram, and more than 86% of cpu time are on FusedConv2D. 
Users could start optimize the top hotspot to improve the performance   
<br><img src="report_template/compare_tf_op_duration_pie.png" width="450" height="300"><br>
</details>
  
<details>
<summary>  Pie Chart for Compare Run Unique TF Ops hotspots  </summary>
  
Users could understand how many percentage this workload spend on unique TF ops only used in the compare run.   
Here is an example diagram, and there is a unique TF ops "PadWithFusedConv2D" token ~10% of total cpu time.       
<br><img src="report_template/unique_compare_tf_op_duration_pie.png" width="450" height="300"><br>
</details>
  
<details>
<summary> Table for Base Run TF ops Elapsed time  </summary>
  
Users could understand exact elpased time for each TF ops from Base run.   
Here is an example diagram.       
<br><img src="report_template/tf_ops_1_dataframe.png" width="800" height="500"><br>
</details> 
  
<details>
<summary> Table for Compare Run TF ops Elapsed time   </summary>
  
Users could understand exact elpased time for each TF ops from Compare run.   
Here is an example diagram.     
<br><img src="report_template/tf_ops_2_dataframe.png" width="800" height="500"><br>
</details>  

<details>
<summary> Table for Compare Run TF ops Elapsed time with shape info  </summary>
  
Users could understand exact elpased time for each TF ops with shape info from Compare run.   
Here is an example diagram.     
<br><img src="report_template/tf_ops_shape_2_dataframe.png" width="800" height="500"><br>
</details>  


<details>
<summary> Table for TF ops Elapsed time comparison   </summary>
  
Users could understand exact elpased time for each TF ops from both run and related speedup number.   
If the TF ops is accelerated with oneDNN, mkl_op would be marked as True.  
If the TF ops is accelerated with native format, native_op would be marked as True.    
Here is an example diagram.  
<br><img src="report_template/common_ops_dataframe.png" width="800" height="500"><br>
</details>
