#!/bin/bash
nvprof --print-gpu-trace -o $1.$OMPI_COMM_WORLD_RANK.nvprof ${@:2}
