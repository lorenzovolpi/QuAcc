#!/bin/bash

SERVER="dgx"
daemon=0
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--host)
            SERVER="$2"
            shift
            shift
            ;;
        -d|--daemon)
            daemon=1
            shift
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

if [[ $SERVER == "dgx" ]]; then
    HOST="lorenzovolpi@edge-nd1.isti.cnr.it"
    WORKDIR="raid/quacc"
elif [[ $SERVER == "bcuda" || $SERVER == "barracuda" ]]; then
    HOST="volpi@barracuda.isti.cnr.it"
    WORKDIR="quacc"
fi

FROM_WORKDIR="$WORKDIR/output"
LOCAL_WORKDIR="/home/lorev/quacc/output"
TO_WORKDIR="/home/volpi/quacc_out"

sync_res() {
    rsync -a "$HOST:$FROM_WORKDIR/results" $LOCAL_WORKDIR
    rsync -a "$LOCAL_WORKDIR/results" "volpi@ilona.isti.cnr.it:$TO_WORKDIR"
}

daemon_sync_res() {
    while true; do
        sync_res
        sleep 10
    done
}

if [[ daemon -gt 0 ]]; then
    daemon_sync_res & disown
else
    sync_res
fi
