#!/bin/bash

script_dir=`dirname $0`

. $script_dir/env.common

python generator.py --nproc `nproc` "$@"
