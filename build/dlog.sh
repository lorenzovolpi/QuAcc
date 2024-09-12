#!/bin/bash

WORKDIR="quacc"


if [[ $1 == "-l" ]]; then
    $host cat $WORKDIR/output/quacc.log | bat -l syslog
else
    $host tail -n +0 -f $WORKDIR/output/quacc.log | bat -P -l syslog
fi
