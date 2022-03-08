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
| Modin*, oneAPI Data Analytics Library (oneDAL), IDP | [Census](Census)       | Use Intel® Distribution of Modin* to ingest and process U.S. census data from 1970 to 2010 in order to build a ridge regression based model to find the relation between education and the total income earned in the US.
| OpenVino™           | [LidarObjectDetection-PointPillars](LidarObjectDetection-PointPillars) | Performs 3D object detection and classification using point cloud data from a LIDAR sensor as input.

# Using Samples in the Intel oneAPI DevCloud
To get started using samples in the DevCloud, refer to [Using AI samples in Intel oneAPI DevCloud](https://github.com/intel-ai-tce/oneAPI-samples/tree/devcloud/AI-and-Analytics#using-samples-in-intel-oneapi-devcloud).

### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.
 - (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the Generate Launch Configurations extension.

To learn more about the extensions, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).