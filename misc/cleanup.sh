#!/bin/bash
# Cleans up auxiliary files, etc. Usage is ./cleanup.sh clean-dir where clean-dir is a path to the directory to clean

if [[ $# > 0 ]]; then
	cd $1
fi
rm -f *.aux *.bcf *.blg *.fdb* *.fls *.log *.out *.xml *.gz

echo Clean!