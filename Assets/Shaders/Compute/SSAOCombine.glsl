//////////////////////////////////////////////////////////////////////
/////                       Compute Shader                      //////
//////////////////////////////////////////////////////////////////////
#version 450 core
layout (local_size_x = 16, local_size_y = 16) in;
layout(rgba8, binding = 0) uniform image2D img_output;

layout(std430, binding=0) buffer Input
{
    vec2 OutputSize;
    float uAOFactor;
};

layout(binding = 0) uniform sampler2D u_Texture;
layout(binding = 1) uniform sampler2D u_SSAO;

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

    vec3 col = DecodeRGBM(texture(u_Texture, vec2(u,v)));
    float ssao = texture(u_SSAO, vec2(u,v)).r;
    ssao = clamp(1 - uAOFactor * (1-ssao), 0, 1);
    pixel = encodeRGBM(col * ssao, kRGBMRange);
    
    // output to a specific pixel in the image
    imageStore(img_output, pixel_coords, pixel);
}
