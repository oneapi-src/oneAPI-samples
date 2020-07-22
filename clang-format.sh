#!/usr/bin/env bash

if [[ -z $CLANG_FORMAT ]] ; then
    CLANG_FORMAT=clang-format
fi

if NOT type $CLANG_FORMAT 2> /dev/null ; then
    echo "No appropriate clang-format found."
    exit 1
fi

FAIL=0
SOURCE_FILES=`find . -type f \( -name '*.h' -name '*.hpp' -o -name '*.cpp' -o -name '*.c' \)`
for i in $SOURCE_FILES
do
    $CLANG_FORMAT -style=llvm $i | grep -c "<replacement " > /dev/null
    if [ $? -ne 1 ]
    then
        echo "$i failed clang-format check."
        FAIL=1
    fi
done
if [ $FAIL -ne 0 ]
then
    echo "Clang-format check failed!"
fi

exit $FAIL

