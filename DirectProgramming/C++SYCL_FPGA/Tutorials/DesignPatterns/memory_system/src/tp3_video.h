#include <fstream>
#include <stdio.h>
#include <stdint.h>
#include <sstream>
#include <iostream>
#include <string> 
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/prototype/pipes_ext.hpp>
#include <sycl/ext/intel/ac_types/ac_fixed.hpp>
#include <sycl/sycl.hpp>
#include <stdlib.h>

#include "exception_handler.hpp"

using namespace std;

#define NB_COLONNE_MAX 1024
#define TAILLE_IM 401

