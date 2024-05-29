#!/bin/bash

python3 main.py fit linear -d homageactions -d_test metaverseactions --class_split --unfreeze_head > output2.txt

echo "Done"
exit 0
