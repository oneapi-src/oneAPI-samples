# Visual Computing Tutorials

Source code area for learning about visual computing development from Intel

## **hellodxtriangle**

 - Additional source code for a technical introduction article to development with the Intel® OSPRay API and the Intel® OSPRay Studio showcase application.
 - Targeted for those familiar with Microsoft* DirectX* and the DirectX* 11 introductory source code.
 - Generates the DirectX* version of comparable scenes to *ospTutorial* and Intel OSPRay Studio scene graph test sources.
- Article Link: TBD

**Try it:**
 - Requirements: 
	 - Windows 10, Microsoft Visual Studio 2019
	 - Windows SDK 10.0.19041.0 (or higher)
	 - DirectX* 11 Toolkit
		 - The toolkit libraries are automatically downloaded through the CMakeLists.txt file with vcpkg.
		 - 600MB on disk.
 - Clone and build with *x64 Native Tools Command Prompt for VS 2019*:
	 - `git clone https://github.com/oneapi-src/oneAPI-samples oneAPI-samples`
	 - `cd oneAPI-samples\VisualComputing\hellodxtriangle`
	 - `mkdir build`
	 - `cd build`
	 - If you have a proxy set it: `set HTTPS_PROXY=http://your.proxy.com:port`
	 - `cmake -G "Visual Studio 16 2019" -A x64 ..`
	 - `cmake --build . --config Release`
 - Run program and open the result image
	 - `cd Release`
	 - `hellodxtriangle.exe`
	 - `explorer hellodxtriangle.png`
