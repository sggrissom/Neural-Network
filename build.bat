@echo off

set IgnoredWarnings= -wd4505 -wd4201 -wd4100 -wd4189
set CommonCompilerFlags= -DSLOW=1 -DDEBUG=1 -I w:\slib -nologo -Gm- -GR- -EHa- -Oi -WX -W4 -FC -Z7 %IgnoredWarnings%
set DebugCompilerFlags= -DSLOW=1 -DDEBUG=1 -MTd -Od %CommonCompilerFlags%
set ReleaseCompilerFlags= -MT -O2 %CommonCompilerFlags%
set CommonLinkerFlags= -incremental:no -opt:ref

IF NOT EXIST bin mkdir bin
pushd bin

cl %DebugCompilerFlags% ..\src\nn.cpp /link %CommonLinkerFlags%

popd
