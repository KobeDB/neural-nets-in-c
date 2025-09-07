#!/bin/bash
set -e

mkdir -p bin

pushd bin
clang -std=c99 -pedantic -D_GNU_SOURCE \
    ../src/main.c -o main -g \
    -I../src \
    -lm
popd
