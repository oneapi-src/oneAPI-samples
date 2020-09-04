# oneDNN verbose log parser


## prerequisites 


*  users need to get a oneDNN verbose log from their workloads first.  

## how to parse logs

### Raw log from frameworks like tensorflow or pytorch
*  parse a raw log "log.txt" from workload : `$profile profile_utils.py log.txt` 
    *  users will see output from console
    *  users will also get some pie chart diagram PNG files like typeTime Breakdown.png
    *  users will also get a parsed output mkldnn_log.csv which only contains onednn logs

### Pure oneDNN log or parsed ouput 'mkldnn_log.csv'
*  parse a onednn log "mkldnn_log.csv" : `$profile profile_utils.py mkldnn_log.csv` 
    * users will see output from console 
    * users will also get some pie chart diagram PNG files like typeTime Breakdown.png

### Compare two pure oneDNN logs 
*  compare two onednn log "a.csv" and "b.csv" : `$profile profile_utils.py a.csv b.csv` 
    * users will see output from console 
    * users will also get a bar chart diagram PNG files like typeTime Comparison.png