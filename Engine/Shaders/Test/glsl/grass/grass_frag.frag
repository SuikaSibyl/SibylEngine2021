#version 450
#extension GL_NV_mesh_shader: require

layout (location = 0) in PerVertexData
{
  vec3 fragColor;
  vec2 fragTexCoord;
} v_out;

layout(location = 0) out vec4 outColor;

layout(binding = 4) uniform sampler2D albedoSampler;

void main() {
  vec4 albedo = texture(albedoSampler, v_out.fragTexCoord);
  if(albedo.a < 0.5) discard;
  outColor = albedo * vec4(v_out.fragColor, 1.0);
}