import numpy as np
import dpctl.tensor as dpt
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.cluster import DBSCAN
import dpctl

def dbscan():
    R = 6
    C = 2
    print("DBScan")
    usm_type = "device"
    device = dpctl.SyclDevice("gpu")
    queue = dpctl.SyclQueue(device)
    device.print_device_info()
    
    X = np.array([[1, 2], [2, 2], [2, 3],
                  [8, 7], [8, 8], [25, 80]], dtype=np.float32)
        
    #X = np.array(np.random.random((R,C)), np.float32)
    # Move data to device memory
    #x_device = dpctl.tensor.from_numpy(X, usm_type = 'device')
    
    x_device = dpt.usm_ndarray(
        X.shape,
        dtype=np.float32,
        buffer=usm_type,
        buffer_ctor_kwargs={"queue": queue},
    )
    # DBSCAN run on the device and result lables also locate in the device memory
    labels_device = DBSCAN(eps=3, min_samples=2).fit_predict(x_device)
    # results can be used in another algorithms or explicitly copied to the host and accessed for output
    #labels_host = dpctl.tensor.to_numpy(labels_device)   
    #print(labels_host)  
    print(x_device)
    
if __name__ == "__main__":
    dbscan()    
     
