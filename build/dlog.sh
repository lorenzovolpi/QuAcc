#!/bin/bash

WORKDIR="quacc"

host="bcuda"
LOGFILE="quacc"

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--module)
            LOGFILE="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

if [[ $1 == "-l" ]]; then
    $host cat "$WORKDIR/output/${LOGFILE}.log" | bat -l syslog
else
    $host tail -n +0 -f "$WORKDIR/output/${LOGFILE}.log" | bat -P -l syslog
fi
