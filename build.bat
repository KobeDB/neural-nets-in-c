@echo off

mkdir bin
pushd bin
cl.exe ..\main.c -Zi
popd bin
