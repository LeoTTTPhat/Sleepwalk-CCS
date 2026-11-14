#!/bin/sh
# Prepare an OSRM (MLD) routing graph for the Austin extract. Run under memguard.
set -e
D=./data/osrm
IMG=ghcr.io/project-osrm/osrm-backend:v5.27.1
docker run --rm -m 6g -v "$D:/data" $IMG osrm-extract -p /opt/car.lua /data/austin.osm.pbf
docker run --rm -m 6g -v "$D:/data" $IMG osrm-partition /data/austin.osrm
docker run --rm -m 6g -v "$D:/data" $IMG osrm-customize /data/austin.osrm
echo OSRM_PREPARED
