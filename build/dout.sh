#!/bin/bash

host="bcuda"
PYTHON_PATH="miniconda3/bin/python"
WORKDIR="quacc"
OUTPUT_DIR="output"
OUTPUT_FILE="quacc"
FILTER=false
TAIL=false


while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tail)
            TAIL=true
            shift 
            ;;
        -m|--module)
            OUTPUT_FILE=$2
            shift
            shift
            ;;
        -d|--dir)
            OUTPUT_DIR=$2
            shift
            shift
            ;;
        -f|--filter)
            FILTER=true
            shift
            ;;
        -*|--*)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

if [[ $TAIL == true ]]; then
    $host tail -n +0 -f "$WORKDIR/$OUTPUT_DIR/${OUTPUT_FILE}.out" | bat -P
else
    if [[ $FILTER == true ]]; then
        $host "$PYTHON_PATH" "$WORKDIR/filter_out.py" "$WORKDIR/$OUTPUT_DIR/${OUTPUT_FILE}.out" | bat
    else
        $host cat "$WORKDIR/$OUTPUT_DIR/${OUTPUT_FILE}.out" | bat
    fi
fi
