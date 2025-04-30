#!/bin/bash
set -e

if [[ "$1" = "serve" ]]; then
    shift 1
    source activate InterpAny
    mkdir /InterpAny-Clearer/webapp/backend/data/results
    cd /InterpAny-Clearer/webapp/backend && nohup python app.py &
    cd /InterpAny-Clearer/webapp/webapp && yarn && nohup yarn start &
    echo "Webapp is running on http://localhost:8080"
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null