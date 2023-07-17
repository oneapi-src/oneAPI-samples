# `Intel® Modin* Vs. Pandas Performance` Sample

The `Intel® Modin* Vs. Pandas Performance` code illustrates how to use Modin* to replace the Pandas API. The sample compares the performance of Intel® Distribution of Modin* and the performance of Pandas for specific dataframe operations.

| Area                       | Description
|:---                        |:---
| What you will learn        | How to accelerate the Pandas API using Intel® Distribution of Modin*.
| Time to complete           | Less than 10 minutes
| Category                   | Concepts and Functionality

## Purpose

Intel® Distribution of Modin* accelerates Pandas operations using Ray or Dask execution engine. The distribution provides compatibility and integration with the existing Pandas code. The sample code demonstrates how to perform some basic dataframe operations using Pandas and Intel® Distribution of Modin*. You will be able to compare the performance difference between the two methods.

You can run the sample locally or in Google Colaboratory (Colab).

## Prerequisites

| Optimized for             | Description
|:---                       |:---
| OS                        | Ubuntu* 20.04 (or newer)
| Hardware                  | Intel® Core™ Gen10 Processor <br> Intel® Xeon® Scalable Performance processors
| Software                  | Intel® AI Analytics Toolkit (AI Kit) <br> Intel® Distribution of Modin*

## Key Implementation Details

This code sample is implemented for CPU using Python programming language. The sample requires NumPy, Pandas, Modin libraries, and the time module in Python.

## Run the `Intel® Modin Vs Pandas Performance` Sample Locally

If you want to run the sample on a local system using a command-line interface (CLI), you must install the Intel® Distribution of Modin* in a new Conda* environment first.

### Install the Intel® Distribution of Modin*

1. Create a Conda environment.
   ```
   conda create --name aikit-modin
   ```
2. Activate the Conda environment.
   ```
   source activate aikit-modin
   ```
3. Remove existing versions of Modin* (if any exist).
   ```
   conda remove modin --y
   ```
4. Install Intel® Distribution of Modin* (v0.12.1 or newer).
   ```
   pip install modin[all]==0.12.1
   ```
5. Install the NumPy and Pandas libraries.
   ```
   pip install numpy
   pip install pandas
   ```
6. Install ipython to run the notebook on your system.
   ```
   pip install ipython
   ```
### Run the Sample

1. Change to the directory containing the `IntelModin_Vs_Pandas.ipynb` notebook file on your local system.

2. Run the sample notebook.
   ```
   ipython IntelModin_Vs_Pandas.ipynb
   ```

## Run the `Intel® Modin Vs Pandas Performance` Sample in Google Colaboratory

1. Change to the directory containing the `IntelModin_Vs_Pandas.ipynb` notebook file on your local system.

2. Open the notebook file, and remove the prepended number sign (#) symbol from the following lines:
   ```
   #!pip install modin[all]==0.12.1
   #!pip install numpy
   #!pip install pandas
   ```
   These changes will install the Intel® Distribution of Modin* and the NumPy and Pandas libraries when run in the Colab notebook.

3. Save your changes.

4. Open [Google Colaboratory](https://colab.research.google.com/?utm_source=scs-index).

5. Sign in to Colab using your Google account.

6. Select **File** > **Upload notebook**.

7. Upload the modified notebook file.

8. Change to the notebook, and click **Open**.

9. Select **Runtime** > **Run all**.

## Example Output

>**Note**: Your output might be different between runs on the notebook depending upon the random generation of the dataset. For the first run, Modin may take longer to execute than Pandas for certain operations since Modin performs some initialization in the first iteration.

```
CPU times: user 8.47 s, sys: 132 ms, total: 8.6 s
Wall time: 8.57 s
```

Example expected cell output is included in `IntelModin_Vs_Pandas.ipynb`.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).