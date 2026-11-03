#!/bin/bash
# run a program of the pgm-omm env on the login node (glibc 2.17) with conda's glibc 2.28 sysroot:
#   glibc228.sh python script.py ...
S=$HOME/miniconda3/pkgs/sysroot_linux-64-2.28-h3108a97_1/x86_64-conda-linux-gnu/sysroot
E=$HOME/miniconda3/envs/pgm-omm
P=$1; shift
[ "$P" = python ] && P=$E/bin/python
exec $S/lib64/ld-2.28.so --library-path $S/lib64:$S/usr/lib64:$E/lib:$E/lib/plugins:$OCL_LIB "$P" "$@"
