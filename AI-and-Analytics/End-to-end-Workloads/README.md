# End-to-end samples for Intel® oneAPI AI Analytics Toolkit (AI Kit)

The Intel® oneAPI AI Analytics Toolkit (AI Kit) gives data scientists, AI
developers, and researchers familiar Python* tools and frameworks to
accelerate end-to-end data science and analytics pipelines on Intel®
architectures. The components are built using oneAPI libraries for low-level
compute optimizations. This toolkit maximizes performance from preprocessing
through machine learning, and provides interoperability for efficient model
development.

You can find more information at
[AI Kit](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html).

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt)
for details.

Third party program Licenses can be found here:
[third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

# End-to-end Samples

| Components         | Folder                 | Description
| ------------------ | ---------------------- | -----------
| Modin, oneDAL, IDP | [Census](Census)       | Use Intel® Distribution of Modin to ingest and process U.S. census data from 1970 to 2010 in order to build a ridge regression based model to find the relation between education and the total income earned in the US.
| OpenVino           | [LidarObjectDetection-PointPillars](LidarObjectDetection-PointPillars) | Performs 3D object detection and classification using point cloud data from a LIDAR sensor as input.

# Using Samples in the Intel oneAPI DevCloud

You can use AI Kit samples in the
[Intel oneAPI DevCloud](https://devcloud.intel.com/oneapi/get-started/)
environment using the following methods:

* Login to a DevCloud system using SSH
* use `git clone` to get a full copy of the oneAPI samples repository
* or use the `oneapi-cli` tool to download this specific sample
* Launch a JupyterLab server and run Jupyter Notebooks from your web browser.
