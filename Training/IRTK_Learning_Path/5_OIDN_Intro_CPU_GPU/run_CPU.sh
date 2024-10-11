#!/bin/bash
source /opt/intel/oneapi/setvars.sh --force &>/dev/null
/bin/echo "##" $(whoami) is running ./bin/oidnDenoise
./bin/oidnDenoise -d cpu \
-hdr /home/common/data/Big_Data/IRTK_BD/OIDN_BD/images/JunkShop_00064spp.hdr.pfm \
-o JunkShop_denoised_CPU.pfm