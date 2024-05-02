#!/usr/bin/env python
import subprocess

subprocess.run(
    ["flatc", "--python", "-o", "lib/flatbuffers", "flatbuffers/npy.fbs"],
    # stdout=subprocess.PIPE,
    # stderr=subprocess.PIPE,
)
subprocess.run(
    ["flatc", "--rust", "-o", "rust/vis/src/flatbuffers", "flatbuffers/npy.fbs"],
    # stdout=subprocess.PIPE,
    # stderr=subprocess.PIPE,
)
