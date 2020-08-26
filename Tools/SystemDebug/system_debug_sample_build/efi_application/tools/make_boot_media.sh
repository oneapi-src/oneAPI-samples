#!/bin/sh

# Copyright (c) 2020, Intel Corporation. All rights reserved.<BR>
# SPDX-License-Identifier: BSD-2-Clause-Patent

# Create a bootable USB thumb drive loaded with a given EFI application.
#
# WARNING WARNING WARNING!!! This will erase the USB drive

if [ $# -ne 2 ] ; then
    echo "USAGE: make-boot-media.sh <.efi file> <USB device e.g. /dev/sdb>"
    exit 1
fi

EFI=$1
DEV=$2

set -e
set -x
sync

if [ "$(uname)" == "Darwin" ]; then
    NAME="TEST"
    sudo gpt destroy /dev/disk2
    sudo gpt create /dev/disk2
    sudo gpt add -i 1 -b 2048 -s 2095104 -t efi $DEV
    sudo newfs_msdos -v $NAME /dev/disk2s1
    sudo diskutil mount /dev/disk2s1

    BOOTDIR=/Volumes/$NAME/EFI/boot
    sudo mkdir -p $BOOTDIR

    sudo cp -v $EFI $BOOTDIR/bootx64.efi
    sync
    sudo diskutil unmountDisk $DEV

elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    sudo parted -s -a minimal $DEV -- \
        mklabel gpt \
        mkpart EFI FAT32 1MiB 1024MiB \
        toggle 1 boot

    sync
    sudo mkfs.vfat "$DEV"1

    sudo parted $DEV -s print

    MOUNTDIR=$(mktemp -d --suffix=_mount_point)
    sudo mount -o umask=000,dmask=000 "$DEV"1 $MOUNTDIR

    BOOTDIR=$MOUNTDIR/EFI/boot
    mkdir -p $BOOTDIR

    cp -v $EFI $BOOTDIR/bootx64.efi

    sync
    sudo umount $MOUNTDIR

    rm -rfv $MOUNTDIR

else
    echo  "Only Mac OS and Linux are supported"
fi
