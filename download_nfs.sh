#!/bin/bash

if [[ ! -d NFS ]]; then
	mkdir NFS
fi

cd NFS

if [[ -z "$(ls -A .)" ]]; then
	curl -fSsl http://ci2cv.net/nfs/Get_NFS.sh | bash -
fi

for filename in *.zip ; do
	if [[ ! -d ${filename%.*} ]]; then
		unzip -j ${filename} ${filename%.*}/240/${filename%.*}/* -d ./${filename%.*}
		rm ${filename}
	fi
done