#!/bin/bash

if [ -z ${NCCS} ]; then
  NCCS=1
fi

if [ -z ${NGPUS} ]; then
  NGPUS=1
fi

if [ -z ${NSTACKS} ]; then
  NSTACKS=1
fi

subdevices=$((NGPU*NSTACK))

export ZE_AFFINITY_MASK=$(((MPI_LOCALRANKID/NCCS)%subdevices))

echo MPI_LOCALRANKID = $MPI_LOCALRANKID  ZE_AFFINITY_MASK = $ZE_AFFINITY_MASK
exec $@
