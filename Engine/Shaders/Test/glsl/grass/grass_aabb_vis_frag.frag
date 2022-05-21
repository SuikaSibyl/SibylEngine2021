#version 450
#extension GL_NV_mesh_shader: require

layout(location = 0) out vec4 outColor;

layout (location = 0) perprimitiveNV in PerPrimitiveData
{
  vec3 fragColor;
} p_out;   // [max_vertices]

void main() {
    outColor = vec4(p_out.fragColor,0.05);
}