import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.cluster import DBSCAN
import dpctl

def dbscan():    
    print("DBScan")
    device = dpctl.SyclDevice("gpu")
    queue = dpctl.SyclQueue(device)    
    
    X = np.array([[1., 2.], [2., 2.], [2., 3.],
                  [8., 7.], [8., 8.], [25., 80.]], dtype=np.float32)
    # Move data to device memory
    x_device = dpctl.tensor.from_numpy(X, usm_type = 'device', 
                                       queue=dpctl.SyclQueue("gpu"))
    # DBSCAN run on the device and result lables also locate in the device memory
    labels_device = DBSCAN(eps=3, min_samples=2).fit_predict(x_device)
    # results can be used in another algorithms or explicitly copied to the host and accessed for output
    labels_host = dpctl.tensor.to_numpy(labels_device)   
       
    print(labels_host)
    
if __name__ == "__main__":
    dbscan()
    
    
    
