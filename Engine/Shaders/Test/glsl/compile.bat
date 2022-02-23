set filename = "aaaaaaaaaaa"
echo filename
echo filename:~0,-4
glslc ./%1 -o ../../../Binaries/Runtime/spirv/%1:~0,-4%spv