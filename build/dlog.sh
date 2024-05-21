#!/bin/bash

host="dgx"

if [[ $# -ge 1 ]]; then
    host=$1
    shift
fi

WORKDIR="quacc"
if [[ $host == "dgx" ]]; then
    WORKDIR="raid/quacc"
fi

mode="log"

if [[ $# -ge 1 ]]; then
    mode=$1
    shift
fi

if [[ $1 == "-l" ]]; then
    if [[ $mode == "log" ]]; then
        $host cat $WORKDIR/output/quacc.log | bat -l syslog
    elif [[ $mode == "out" ]]; then
        $host cat $WORKDIR/output/quacc.out | bat
    fi
else
    if [[ $mode == "log" ]]; then
        $host tail -n +0 -f $WORKDIR/output/quacc.log | bat -P -l syslog
    elif [[ $mode == "out" ]]; then
        $host tail -n +0 -f $WORKDIR/output/quacc.out | bat -P
    fi
fi