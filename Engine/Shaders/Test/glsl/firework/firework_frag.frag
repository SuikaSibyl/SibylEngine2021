#version 450
#extension GL_NV_mesh_shader: require

layout (location = 0) in PerVertexData
{
  vec3 fragColor;
  vec2 fragTexCoord;
} v_out;   // [max_vertices]

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(v_out.fragColor,1);
}