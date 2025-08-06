@echo off

mkdir bin_scratch
pushd bin_scratch
cl.exe ..\scratch\scratch_main.c -Zi
popd bin_scratch
