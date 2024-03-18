#!/usr/bin/env bash
set -x
set -e
cat src/hpshader.glsl | glslc -DVERTEX -fshader-stage=vertex -fentry-point=vertex_main -o hpshader.vertex.spv -
cat src/hpshader.glsl | glslc -DFRAGMENT -fshader-stage=fragment -fentry-point=fragment_main -o hpshader.fragment.spv -
