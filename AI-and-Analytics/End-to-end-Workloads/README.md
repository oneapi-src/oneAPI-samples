# End-to-End Samples for the AI Tools

The AI Tools allows data scientists, AI
developers, and researchers familiar Python* tools and frameworks to
accelerate end-to-end data science and analytics pipelines on Intel®
architectures. The components are built using oneAPI libraries for low-level
compute optimizations.

The AI Tools maximizes performance from preprocessing
through machine learning, and provides interoperability for efficient model
development.

You can find more information at
[AI Tools](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html).


# End-to-end Samples

| Components         | Folder                 | Description
| :---               |:---                    |:---
| Intel® Distribution of Modin* <br> Intel® oneAPI Data Analytics Library (oneDAL) <br> IDP | [Census](Census)       | Use Intel® Distribution of Modin* to ingest and process U.S. census data from 1970 to 2010 in order to build a ridge regression based model to find the relation between education and the total income earned in the US.
| Intel Extension for PyTorch (IPEX), Intel Neural Compressor (INC)           | [LanguageIdentification](LanguageIdentification) | Trains a model to perform language identification using the Hugging Face Speechbrain library and CommonVoice dataset, and optimized with IPEX and INC.
| Intel® Distribution of OpenVINO™ toolkit           | [LidarObjectDetection-PointPillars](LidarObjectDetection-PointPillars) | Performs 3D object detection and classification using point cloud data from a LIDAR sensor as input.


## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).
