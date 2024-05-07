#!/bin/bash

mode="log"

if [[ $# -ge 1 ]]; then
    mode=$1
    shift
fi

if [[ $1 == "-l" ]]; then
    if [[ $mode == "log" ]]; then
        dgx cat raid/quacc/output/quacc.log | bat -l syslog
    elif [[ $mode == "out" ]]; then
        dgx cat raid/quacc/output/quacc.out | bat
    fi
else
    if [[ $mode == "log" ]]; then
        dgx tail -n +0 -f raid/quacc/output/quacc.log | bat -P -l syslog
    elif [[ $mode == "out" ]]; then
        dgx tail -n +0 -f raid/quacc/output/quacc.out | bat -P
    fi
fi