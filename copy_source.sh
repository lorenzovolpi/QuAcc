#!/bin/bash

# CMD="cp"
# DEST="~/tesi_docker/"
CMD="scp"
DEST="andreaesuli@edge-nd1.isti.cnr.it:/home/andreaesuli/raid/lorenzo/"

bash -c "${CMD} -r quacc ${DEST}"
bash -c "${CMD} -r baselines ${DEST}"
bash -c "${CMD} run.py ${DEST}"
bash -c "${CMD} remote.py ${DEST}"
bash -c "${CMD} conf.yaml ${DEST}"
bash -c "${CMD} requirements.txt ${DEST}"
