{
  "guid": "AA458E3A-932C-460E-97A7-5962AF0C41FA",
  "name": "TensorFlow Multinode Training with Horovod",
  "categories": ["Toolkit/Intel® AI Analytics Toolkit/TensorFlow"],
  "description": "This sample shows how to train a TensorFlow and run inference with oneMKL and oneDNN.",
  "builder": ["cli"],
  "languages": [{"python":{}}],
  "os":["linux"],
  "ciTests": {
	"linux": [
	{
		"id": "tensorflow horovod",
		"steps": [
			"source activate tensorflow",
			"mpirun -n 2 python TensorFlow_Multinode_Training_with_Horovod.py"
		 ]
	}
     ]
 }
}