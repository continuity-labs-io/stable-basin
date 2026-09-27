#!/bin/bash
# Report GPU idle residency over time

sudo powermetrics --samplers gpu_power -i 4000 | awk '/GPU idle residency:/ {
    val = $4
    sub("%", "", val)
    printf "%6.1f%%\n", val
    fflush(stdout)
}'
