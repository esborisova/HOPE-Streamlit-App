#!/bin/bash

file_names=('genåb'
            'restriktion'
            '17jan'
            'corona'
            'coronapas'
            'lockdown'
            'mettef'
            'mundbind'
            'omicron'
            'pressekonference'
            'vaccin');

            


for my_file_name in "${file_names[@]}"; do
    IFS=' ' read f_type func  <<< $my_file_name

    echo "Run: " $my_file_name
    python3 collect_mean_arrays.py $my_file_name

done

