@echo off

set IgnoredWarnings= -wd4505 -wd4201 -wd4100 -wd4189
set CommonCompilerFlags= -DSLOW=1 -DDEBUG=1 -MTd -nologo -Gm- -GR- -EHa- -Od -Oi -WX -W4 -FC -Z7 %IgnoredWarnings%
set CommonLinkerFlags= -incremental:no -opt:ref

IF NOT EXIST bin mkdir bin
pushd bin

cl %CommonCompilerFlags% ..\src\nn.cpp /link %CommonLinkerFlags%

popd
