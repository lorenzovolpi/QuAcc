#!/bin/bash

SERVER="dgx"
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--host)
            SERVER="$2"
            shift
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
fi


IMAGE_NAME="lorenzovolpi/quacc_main"
DEST="$HOST:$WORKDIR"

rsync -ai quacc playground $DEST
poetry export --without-hashes --with=dev,dash --format=requirements.txt 2>/dev/null 1>requirements.txt
rsync -i docker/Dockerfile requirements.txt docker/drun.sh docker/ddebug.sh docker/dbuild.sh docker/docker.env docker/.dockerignore $DEST
ssh $HOST "cd $WORKDIR; ./dbuild.sh $IMAGE_NAME"

