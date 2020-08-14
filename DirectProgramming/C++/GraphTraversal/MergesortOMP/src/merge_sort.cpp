//==============================================================
//
// SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
// http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
//
// Copyright Intel Corporation
//
// THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
//
// =============================================================
#define _CRT_SECURE_NO_DEPRECATE
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <iostream>

constexpr int task_threshold = 5000;
constexpr int n = 100000000;

// Description:
// Initializes the array, and shuffle all elements in it.
//
// [in]:  a       Array to be initialized.
//        n       Length of array.
// [out]: None.
void InitializeArray(int a[], int n) {
  for (int i = 0; i < n; ++i) a[i] = i;

  printf("Shuffling the array\n");
  std::random_shuffle(a, a + n);
}

// Description:
// Checks that array is sorted and each element contains the index.
//
// [in]:  a       Array to be initialized.
//        n       Length of array.
// [out]: Return 0 if no error is found. Otherwise, return 1.
int CheckArray(int a[], int n) {
  for (int i = 0; i < n - 1; ++i) {
    if (a[i] >= a[i + 1] || a[i] != i) {
      printf("Sort failed at location %d, a[i]=%d a[i+1]=%d\n", i, a[i],
             a[i + 1]);
      return 1;
    }
  }

  // no error
  return 0;
}

// Description:
// Merges two sublists of array
//
// [in]:  a       Array to be sorted.
//        tmp_a   Temporary array to contain sorted numbers.
//        first   Index of first element of first sublist to be merged in array
//        a. middle  Index of first element of second sublist to be merged in
//        array a. last    Index of last element of second sublist to be merged
//        in array a.
// [out]: a
void Merge(int a[], int tmp_a[], int first, int middle, int last) {
  // Merge two portions of array, and put result into a temporary array tmp_a[]
  int p1 = first;
  int p2 = middle;
  int p = first;
  while (p <= last) {
    if (p1 < middle && (p2 > last || a[p1] < a[p2])) {
      tmp_a[p++] = a[p1++];
    } else {
      tmp_a[p++] = a[p2++];
    }
  }

  // Copy sorted portion from the temporary array to the original array
  for (int i = first; i <= last; ++i) {
    a[i] = tmp_a[i];
  }
}

// Description:
// Sort the list starting from first to last.
// Use the Merge Sort algorithm, using recursive divide and conquer.
//
// [in]:  a       Array to be sorted.
//        tmp_a   Temporary array to contain sorted sublists.
//        first   Index of first element of the list to be sorted in array a.
//        last    Index of last element of the list to be sorted in array a.
// [out]: a
void MergeSort(int a[], int tmp_a[], int first, int last) {
  if (first < last) {
    int middle = (first + last + 1) / 2;  // = first + (last - first + 1) / 2;
    // Splits list a[first:last] into two halves (called sublists).
    // One is [first:middle-1], and another is [middle:last].
    MergeSort(a, tmp_a, first, middle - 1);
    MergeSort(a, tmp_a, middle, last);
    Merge(a, tmp_a, first, middle, last);
  }
}

// Description:
// OpenMP Task version of merge_sort
void MergeSortOpenMP(int a[], int tmp_a[], int first, int last) {
  if (first < last) {
    int middle = (first + last + 1) / 2;  // = first + (last - first + 1) / 2;
    // Splits list a[first:last] into two halves (called sublists).
    // One is [first:middle-1], and another is [middle:last].
    // For sake of performance, only when the list is big enough,
    // we create tasks with #pragma omp task.
    if (last - first < task_threshold) {
      MergeSort(a, tmp_a, first, middle - 1);
      MergeSort(a, tmp_a, middle, last);
    } else {
#pragma omp task
      MergeSortOpenMP(a, tmp_a, first, middle - 1);
#pragma omp task
      MergeSortOpenMP(a, tmp_a, middle, last);
#pragma omp taskwait
    }
    Merge(a, tmp_a, first, middle, last);
  }
}

int main(int argc, char *argv[]) {
  std::chrono::time_point<std::chrono::system_clock> start1, start2, end1, end2;
  std::chrono::duration<double> elapsed_seconds_serial, elapsed_seconds_openmp;
  printf("N = %d\n", n);

  int *a = new int[n];
  int *tmp_a = new int[n];

  int option = 0;
  // If PERF_NUM is defined, then no options taken...run all tests
#ifndef PERF_NUM
  // Checks to see if option was given at command line
  if (argc > 1) {
    // Prints out instructions and quits
    if (argv[1][0] == 'h') {
      printf("Merge Sort Sample\n");
      printf("[0] all tests\n[1] serial\n[2] OpenMP Task\n");
#ifdef _WIN32
      system("PAUSE");
#endif  // _WIN32
      return 0;
    }
    // option is assumed an option
    else {
      option = atoi(argv[1]);
    }
  }
  // If no options are given, prompt user to choose an option
  else {
    printf("Merge Sort Sample\n");
    printf("[0] all tests\n[1] serial\n[2] OpenMP Task\n");
    scanf("%i", &option);
  }
#else   // !PERF_NUM

  //#ifdef PERF_NUM
  double avg_time[2];
#endif  // PERF_NUM

  switch (option) {
    case 0:
      printf("\nRunning all tests\n");
#ifdef PERF_NUM
      for (int i = 0; i < 5; ++i) {
#endif  // PERF_NUM
        printf("\nSerial version:\n");
        InitializeArray(a, n);
        printf("Sorting\n");
        start1 = std::chrono::system_clock::now();
        MergeSort(a, tmp_a, 0, n - 1);
        end1 = std::chrono::system_clock::now();
        elapsed_seconds_serial = end1 - start1;
        // Confirm that a is sorted and that each element contains the index.
        if (CheckArray(a, n)) {
          delete[] tmp_a;
          delete[] a;
          return 1;
        }
        std::cout << "Sort succeeded in " << elapsed_seconds_serial.count()
                  << " seconds.\n";
        std::cout << "\nOpenMP Task Version:\n";
        InitializeArray(a, n);
        printf("Sorting\n");
        start2 = std::chrono::system_clock::now();
#pragma omp parallel
        {
#pragma omp single
          { MergeSortOpenMP(a, tmp_a, 0, n - 1); }
        }
        end2 = std::chrono::system_clock::now();
        elapsed_seconds_openmp = end2 - start2;

        // Confirm that a is sorted and that each element contains the index.
        if (CheckArray(a, n)) {
          delete[] tmp_a;
          delete[] a;
          return 1;
        }
        std::cout << "Sort succeeded in " << elapsed_seconds_openmp.count()
                  << " seconds.\n";
#ifdef PERF_NUM
        avg_time[0] += elapsed_seconds_serial.count();
        avg_time[1] += elapsed_seconds_openmp.count();
      }
      printf("\n");
      printf("avg time of serial version: %.0fms\n",
             avg_time[0] * 1000.0 / 5);
      printf("avg time of OpenMP Task version: %.0fms\n",
             avg_time[1] * 1000.0 / 5);
#endif  // PERF_NUM
      break;

    case 1:
      printf("\nSerial version:\n");
      InitializeArray(a, n);
      printf("Sorting\n");
      start1 = std::chrono::system_clock::now();
      MergeSort(a, tmp_a, 0, n - 1);
      end1 = std::chrono::system_clock::now();
      elapsed_seconds_serial = end1 - start1;
      // Confirm that a is sorted and that each element contains the index.
      if (CheckArray(a, n)) {
        delete[] tmp_a;
        delete[] a;
        return 1;
      }
      std::cout << "Sort succeeded in " << elapsed_seconds_serial.count()
                << " seconds.\n";
      break;

    case 2:
      printf("\nOpenMP version:\n");
      InitializeArray(a, n);
      printf("Sorting\n");
      start1 = std::chrono::system_clock::now();
#pragma omp parallel
      {
#pragma omp single
        { MergeSortOpenMP(a, tmp_a, 0, n - 1); }
      }
      end1 = std::chrono::system_clock::now();

      elapsed_seconds_openmp = end1 - start1;
      // Confirm that a is sorted and that each element contains the index.
      if (CheckArray(a, n)) {
        delete[] tmp_a;
        delete[] a;
        return 1;
      }
      std::cout << "Sort succeeded in " << elapsed_seconds_openmp.count()
                << " seconds.\n";
      break;

    default:
      printf("Please pick a valid option\n");
      break;
  }

  delete[] tmp_a;
  delete[] a;
  return 0;
}
