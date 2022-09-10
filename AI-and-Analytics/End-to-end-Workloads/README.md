# End-to-End Samples for the Intel® AI Analytics Toolkit (AI Kit)

The Intel® AI Analytics Toolkit (AI Kit) allows data scientists, AI
developers, and researchers familiar Python* tools and frameworks to
accelerate end-to-end data science and analytics pipelines on Intel®
architectures. The components are built using oneAPI libraries for low-level
compute optimizations.

The AI Toolkit maximizes performance from preprocessing
through machine learning, and provides interoperability for efficient model
development.

You can find more information at
[Intel® AI Analytics Toolkit (AI Kit)](https://software.intel.com/content/www/us/en/develop/tools/oneapi/ai-analytics-toolkit.html).


# End-to-end Samples

| Components         | Folder                 | Description
| :---               |:---                    |:---
| Intel® Distribution of Modin* <br> Intel® oneAPI Data Analytics Library (oneDAL) <br> IDP | [Census](Census)       | Use Intel® Distribution of Modin* to ingest and process U.S. census data from 1970 to 2010 in order to build a ridge regression based model to find the relation between education and the total income earned in the US.
| Intel® Distribution of OpenVINO™ toolkit           | [LidarObjectDetection-PointPillars](LidarObjectDetection-PointPillars) | Performs 3D object detection and classification using point cloud data from a LIDAR sensor as input.

# Using Samples in Intel® DevCloud
To get started using samples in the Intel® DevCloud, refer to [*Using AI samples in Intel oneAPI DevCloud*](https://github.com/intel-ai-tce/oneAPI-samples/tree/devcloud/AI-and-Analytics#using-samples-in-intel-oneapi-devcloud).


### Use Visual Studio Code* (VS Code) (Optional)
You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 1. Configure the oneAPI environment with the extension **Environment Configurator for Intel® oneAPI Toolkits**.
 2. Download a sample using the extension **Code Sample Browser for Intel® oneAPI Toolkits**.
 3. Open a terminal in VS Code (**Terminal > New Terminal**).
 4. Run the sample in the VS Code terminal as you would on a Linux* system.
 5. (Linux only) Debug your GPU application with GDB for Intel® oneAPI toolkits using the Generate Launch Configurations extension.

To learn more about the extensions and how to configure the oneAPI environment, see the 
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).


## License

Code samples are licensed under the MIT license. See [License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third-party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).