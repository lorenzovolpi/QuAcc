#!/bin/bash

CMD="scp"
DEST="andreaesuli@edge-nd1.isti.cnr.it:~/raid/lorenzo/"
# CMD="cp"
# DEST="~/tesi_docker/"

bash -c "${CMD} -r quacc ${DEST}"
bash -c "${CMD} -r baselines ${DEST}"
bash -c "${CMD} run.py ${DEST}"
bash -c "${CMD} remote.py ${DEST}"
bash -c "${CMD} conf.yaml ${DEST}"
bash -c "${CMD} requirements.txt ${DEST}"
