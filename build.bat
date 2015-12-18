@echo off

set debug= 0
set release= 1

set IgnoredWarnings= -wd4505 -wd4201 -wd4100 -wd4189
set CommonCompilerFlags= -DSLOW=1 -DDEBUG=1 /openmp -I w:\slib -EHsc -nologo -Gm- -GR- -EHa- -Oi -WX -W4 -FC -Z7 %IgnoredWarnings%
set DebugCompilerFlags= -DSLOW=1 -DDEBUG=1 -MTd -Od -Wv:18 %CommonCompilerFlags%
set ReleaseCompilerFlags= -MT -O2 %CommonCompilerFlags%
set CommonLinkerFlags= -incremental:no -opt:ref

IF %debug%==1 (set CompilerFlags= %DebugCompilerFlags%)
IF %release%==1 (set CompilerFlags= %ReleaseCompilerFlags%)

IF NOT EXIST bin mkdir bin
pushd bin

cl -Iw:\slib %CompilerFlags% ..\src\nn.cpp /link %CommonLinkerFlags%

popd
