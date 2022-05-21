#version 450
#extension GL_NV_mesh_shader: require
#extension GL_NV_fragment_shader_barycentric: require

layout (location = 0) in PerVertexData
{
  vec2 fragTexCoord;
} v_out;   // [max_vertices]
// layout (location = 0) pervertexNV in PerVertexData
// {
//   vec2 fragTexCoord;
// } v_out[];   // [max_vertices]

layout (location = 1) perprimitiveNV in PerPrimitiveData
{
  vec3 fragColor;
} p_out;   // [max_vertices]

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) out vec4 outColor;

uvec2 uv2uuv()
{
  vec2 uv = v_out.fragTexCoord * 4; //[0,4]
  return uvec2(min(uint(uv.x), 3), min(uint(uv.y), 3));
}

float nrand()
{
  float n = v_out.fragTexCoord.x * 10 + v_out.fragTexCoord.y;
	return fract(sin(91.2228 * n)* 43758.5453);
}

float unpackSimpTex(in uint x, in uint y)
{
  uint index = x * 4 + y;
  uint halftex = 0;
  if(index < 8) halftex = floatBitsToUint(p_out.fragColor.x);
  else { halftex = floatBitsToUint(p_out.fragColor.y); index-= 8; }

  uint texel = (halftex >> (28-(index<<2))) & 0xF;
  return float(texel)/15;
}

void main() {
    if(p_out.fragColor.b == -1)
    {
      uvec2 uuv = uv2uuv();
      float rejUnpackCol = unpackSimpTex(uuv.x, uuv.y);
      if(rejUnpackCol > 0) outColor = vec4(0.43529, 0.5098, 0.38039, 1);
      else discard;
    }
    else
    {
      vec4 texture_color = texture(texSampler, v_out.fragTexCoord);
      texture_color.rgb *= p_out.fragColor;
      if(texture_color.a < 0.5) discard;
      outColor = texture_color;
    }
}