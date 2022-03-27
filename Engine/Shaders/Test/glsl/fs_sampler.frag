#version 450
#extension GL_NV_mesh_shader: require

layout (location = 0) in PerVertexData
{
  vec2 fragTexCoord;
} v_out;   // [max_vertices]

layout (location = 1) perprimitiveNV in PerPrimitiveData
{
  vec3 fragColor;
} p_out;   // [max_vertices]

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) out vec4 outColor;

void main() {
    vec4 texture_color = texture(texSampler, v_out.fragTexCoord);
    texture_color.rgb *= p_out.fragColor;
    outColor = texture_color;
}