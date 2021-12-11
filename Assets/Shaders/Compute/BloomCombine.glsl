//////////////////////////////////////////////////////////////////////
/////                       Compute Shader                      //////
//////////////////////////////////////////////////////////////////////
#version 450 core
layout (local_size_x = 16, local_size_y = 16) in;
layout(rgba8, binding = 0) uniform image2D img_output;

layout(std430, binding=0) buffer Input
{
    vec2 OutputSize;
    float uBloomFactor;
};

layout(binding = 0) uniform sampler2D u_Texture;
layout(binding = 1) uniform sampler2D TextureBloomBlur1;
layout(binding = 2) uniform sampler2D TextureBloomBlur2;
layout(binding = 3) uniform sampler2D TextureBloomBlur3;
layout(binding = 4) uniform sampler2D TextureBloomBlur4;
layout(binding = 5) uniform sampler2D TextureBloomBlur5;

const float kRGBMRange = 8.0;

vec4 encodeRGBM(const in vec3 color, const in float range) {
    if(range <= 0.0) return vec4(color, 1.0);
    vec4 rgbm;
    vec3 col = color / range;
    rgbm.a = clamp( max( max( col.r, col.g ), max( col.b, 1e-6 ) ), 0.0, 1.0 );
    rgbm.a = ceil( rgbm.a * 255.0 ) / 255.0;
    rgbm.rgb = col / rgbm.a;
    return rgbm;
}

vec3 DecodeRGBM(vec4 rgbm)
{
    return rgbm.xyz * rgbm.w * kRGBMRange;
}

float getLuminance(const in vec3 color) {
    const vec3 colorBright = vec3(0.2126, 0.7152, 0.0722);
    return dot(color, colorBright);
}

float getRadiusFactored(const float value, const float middle) {
    return mix(value, middle * 2.0 - value, 1);
}

vec4 bloomCombine(vec2 uv) {
    vec3 bloom = vec3(0.0);
    const float midVal = 0.6;
    const float factor1 = 1.1;
    const float factor2 = 0.9;
    const float factor3 = 0.6;
    const float factor4 = 0.3;
    const float factor5 = 0.1;
    bloom += (DecodeRGBM(texture(TextureBloomBlur1, uv))) * getRadiusFactored(factor1, midVal);
    bloom += (DecodeRGBM(texture(TextureBloomBlur2, uv))) * getRadiusFactored(factor2, midVal);
    bloom += (DecodeRGBM(texture(TextureBloomBlur3, uv))) * getRadiusFactored(factor3, midVal);
    bloom += (DecodeRGBM(texture(TextureBloomBlur4, uv))) * getRadiusFactored(factor4, midVal);
    bloom += (DecodeRGBM(texture(TextureBloomBlur5, uv))) * getRadiusFactored(factor5, midVal);
    vec3 color = DecodeRGBM(texture(u_Texture, uv));
    return vec4(color + (bloom * uBloomFactor), 1.0);
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

    vec4 col = bloomCombine(vec2(u,v));
    col = encodeRGBM(col.rgb, kRGBMRange);
    pixel= col;
    
    // output to a specific pixel in the image
    imageStore(img_output, pixel_coords, pixel);
}
