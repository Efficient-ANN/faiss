#!/bin/bash
argList=()
i=1
for var in "${@:2}"
do
    if [[ -z $var ]]; then
        echo "$i = \"\""
        argList+=("")
    else
        echo "$i = $var"
        argList+=($var)
    fi
    i=$((i+1))
done

nvprof --print-gpu-trace -o $1.$OMPI_COMM_WORLD_RANK.nvprof ${argList[@]+"${argList[@]}"}
