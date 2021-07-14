// Solution Using Target Data
#pragma omp target data map(tofrom : x)
{
#pragma omp target map(to : y)
  {
    for (int i = 0; i < ARRAY_SIZE; i++) x[i] += y[i];
  }

  init2(y, ARRAY_SIZE);

#pragma omp target map(to : y)
  {
    for (int i = 0; i < ARRAY_SIZE; i++) x[i] += y[i];
  }
}


// Solution Using Target Enter/Exit/Update
#pragma omp target enter data map(to : x) map(to : y)
#pragma omp target
{
  for (int i = 0; i < ARRAY_SIZE; i++) x[i] += y[i];
}

init2(y, ARRAY_SIZE);

#pragma omp target update to(y)

#pragma omp target
{
  for (int i = 0; i < ARRAY_SIZE; i++) x[i] += y[i];
}
#pragma omp target exit data map(from : x)