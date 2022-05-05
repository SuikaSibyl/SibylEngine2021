set input=%1
set filename=%input:~0,-5%
glslc ./%1 -o ../../../Binaries/Runtime/spirv/%filename%.spv --target-env=vulkan1.1 -g
