Push-Location $PSScriptRoot

cd zlib
cmake .
msbuild /p:Configuration=Debug /p:Platform=x64 .\ALL_BUILD.vcxproj
msbuild /p:Configuration=Release /p:Platform=x64 .\ALL_BUILD.vcxproj

cd ../libpng
cmake -DPNG_BUILD_ZLIB=on .
msbuild /p:Configuration=Debug /p:Platform=x64 /p:IncludePath="..\zlib\" .\png_static.vcxproj
msbuild /p:Configuration=Release /p:Platform=x64 /p:IncludePath="..\zlib\" .\png_static.vcxproj

cd ../yaml-cpp
cmake -DYAML_BUILD_SHARED_LIBS=off .
msbuild /p:Configuration=Debug /p:Platform=x64 .\YAML_CPP.sln
msbuild /p:Configuration=Release /p:Platform=x64 .\YAML_CPP.sln

Pop-Location