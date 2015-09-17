@echo off

set flat=1
set simple=0
set simd=0

set debug=1
set fast=0
set release=0

set IgnoredWarnings= -wd4505 -wd4201 -wd4100 -wd4189
set CommonCompilerFlags= -DSLOW=1 -DDEBUG=1 -I w:\slib -nologo -Gm- -GR- -EHa- -Oi -WX -W4 -FC -Z7 %IgnoredWarnings%
set DebugCompilerFlags= -DSLOW=1 -DDEBUG=1 -MTd -Od %CommonCompilerFlags%
set ReleaseCompilerFlags= -MT -O2 %CommonCompilerFlags%
set CommonLinkerFlags= -incremental:no -opt:ref

IF %debug%==1 (set CompilerFlags= %DebugCompilerFlags%)
IF %fast%==1 (set CompilerFlags= %FastCompilerFlags%)
IF %release%==1 (set CompilerFlags= %ReleaseCompilerFlags%)

IF %flat%==1 (set CompilerFlags= -DFLAT_NN=1 %CompilerFlags%
IF %simple%==1 (set CompilerFlags= -DSIMPLE_NN=1 %CompilerFlags%
IF %simd%==1 (set CompilerFlags= -DSIMD_NN=1 %CompilerFlags%

IF NOT EXIST bin mkdir bin
pushd bin

cl %CompilerFlags% ..\src\nn.cpp /link %CommonLinkerFlags%

popd
