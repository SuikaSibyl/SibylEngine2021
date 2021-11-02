//////////////////////////////////////////////////////////////////////
/////                       Compute Shader                      //////
//////////////////////////////////////////////////////////////////////
#version 450 core
layout (local_size_x = 16, local_size_y = 16) in;
layout(rgba8, binding = 0) uniform image2D img_output;

layout(std430, binding=0) buffer Input
{
    vec2 OutputSize;
    float Para;
};

uniform sampler2D u_Texture;

vec3 ACESToneMapping(vec3 color, float adapted_lum)
{
	const float A = 2.51f;
	const float B = 0.03f;
	const float C = 2.43f;
	const float D = 0.59f;
	const float E = 0.14f;

	color *= adapted_lum;
	return (color * (A * color + B)) / (color * (C * color + D) + E);
}

const float kRGBMRange = 8.0;
vec3 DecodeRGBM(vec4 rgbm)
{
    return rgbm.xyz * rgbm.w * kRGBMRange;
}

vec3 linearTosRGB(const in vec3 color) {
    return vec3( color.r < 0.0031308 ? color.r * 12.92 : 1.055 * pow(color.r, 1.0/2.4) - 0.055, color.g < 0.0031308 ? color.g * 12.92 : 1.055 * pow(color.g, 1.0/2.4) - 0.055, color.b < 0.0031308 ? color.b * 12.92 : 1.055 * pow(color.b, 1.0/2.4) - 0.055);
}

void main(void)
{
    // base pixel colour for image
    vec4 pixel = vec4(1.0, 0.0, 1.0, 1.0);
    // get index in global work group i.e x,y position
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    if(gl_GlobalInvocationID.x >= OutputSize.x || gl_GlobalInvocationID.y >= OutputSize.y)
        return;

    float du = 1.0f / OutputSize.x;
    float dv = 1.0f / OutputSize.y;

    float u = du * (gl_GlobalInvocationID.x + 0.5f);
    float v = dv * (gl_GlobalInvocationID.y + 0.5f);

    vec4 textCol = texture(u_Texture, vec2(u,v));
    vec3 hdrCol = DecodeRGBM(textCol);
    hdrCol = ACESToneMapping(hdrCol, Para);
    hdrCol = linearTosRGB(hdrCol);
    pixel= vec4(hdrCol, 1);

    // output to a specific pixel in the image
    imageStore(img_output, pixel_coords, pixel);
}
