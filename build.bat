@echo off

mkdir bin
pushd bin
cl.exe ..\src\main.c -I..\src -Zi /std:c11
popd bin
