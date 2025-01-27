#!/bin/bash

WORKDIR="quacc"

host="bcuda"
LOGFILE="quacc"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--host)
            host="$2"
            shift
            shift
            ;;
        --local)
            host=""
            WORKDIR="."
            shift
            ;;
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
    $host tail -f -n +0 "$WORKDIR/output/${LOGFILE}.log" | bat -P -l syslog
fi
