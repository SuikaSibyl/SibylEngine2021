#version 450

layout(location = 0) in float depth;
layout(location = 0) out vec4 outColor;

vec4 packFloat32AsVec4(float value)
{
    // uint value_bits = float(value);
    float value_uniform = (value + 1) / 2;
    uint value_bits = uint(value_uniform * float(0xFFFFFFFF));
    float r = (1.f * ((value_bits >> 24) & 255)) / 255;
    float g = (1.f * ((value_bits >> 16) & 255)) / 255;
    float b = (1.f * ((value_bits >> 8) & 255)) / 255;
    float a = (1.f * ((value_bits >> 0) & 255)) / 255;
    return vec4(r,g,b,a);
}

vec4 EncodeFloatRGBA( float v )
{
	vec4 kEncodeMul = vec4(1.0, 255.0, 65025.0, 16581375.0);
	float kEncodeBit = 1.0/255.0;
	vec4 enc = kEncodeMul * v;
	enc = fract(enc);
	enc -= enc.yzww * kEncodeBit;
	return enc;
}

void main() {
    outColor = EncodeFloatRGBA(depth);
}
