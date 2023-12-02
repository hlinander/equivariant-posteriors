#!/bin/bash
set -x
singularity shell --nv --no-home --env PYTHONNOUSERSITE=1 $ENTVAR/equivariant-posteriors/image.img bash
