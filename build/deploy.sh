#!/bin/bash

SERVER="bcuda"
while [[ $# -gt 0 ]]; do
  case $1 in
  -h | --host)
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

  IMAGE_NAME="lorenzovolpi/quacc_main"
  DEST="$HOST:$WORKDIR"

  rsync -ai quacc playground exp $DEST
  poetry export --without-hashes --with=dev,dash --format=requirements.txt 2>/dev/null 1>requirements.txt
  rsync -i build/Dockerfile requirements.txt build/drun.sh build/ddebug.sh build/dbuild.sh build/docker.env build/.dockerignore $DEST
  ssh $HOST "cd $WORKDIR; ./dbuild.sh $IMAGE_NAME"
elif [[ $SERVER == "bcuda" || $SERVER == "barracuda" ]]; then
  HOST="volpi@barracuda.isti.cnr.it"
  WORKDIR="quacc"

  DEST="$HOST:$WORKDIR"

  rsync -ai quacc playground exp $DEST
  rsync -i README.md build/run.sh build/bcuda.env build/filter_out.py pyproject.toml $DEST
  rsync -ai qcdash $DEST
  rsync -i build/dash.sh build/dash.env $DEST
fi
