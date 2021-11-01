//////////////////////////////////////////////////////////////////////
/////                       Compute Shader                      //////
//////////////////////////////////////////////////////////////////////
#version 450 core
layout (local_size_x = 1, local_size_y = 1) in;
layout(rgba8, binding = 0) uniform image2D img_output;

layout(std430, binding=0) buffer Input
{
    vec2 uGlobalTexSize;
    vec2 uTextureBlurInputSize;
    vec2 uBlurDir;
};

layout(binding = 0) uniform sampler2D u_Input;

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

vec4 gaussianBlur(vec2 uv) {
    vec2 maxCoord = 1.0 - 1.0 / uTextureBlurInputSize.xy;
    vec2 minCoord = 1.0 / uTextureBlurInputSize.xy;

    vec3 pixel = 0.2734375 * DecodeRGBM(texture(u_Input, max(min(uv, maxCoord), minCoord)));

    vec2 offset;
    vec2 blurDir = uBlurDir.xy / gl_NumWorkGroups.xy;

    blurDir *= uGlobalTexSize.y * 0.00075;
    offset = blurDir * 1.3333333333333333;
    pixel += 0.328125 *  DecodeRGBM(texture(u_Input, clamp(uv + offset, minCoord, maxCoord)));
    pixel += 0.328125 *  DecodeRGBM(texture(u_Input, clamp(uv - offset, minCoord, maxCoord)));
    offset = blurDir * 3.111111111111111;
    pixel += 0.03515625 * DecodeRGBM(texture(u_Input, clamp(uv + offset, minCoord, maxCoord)));
    pixel += 0.03515625 * DecodeRGBM(texture(u_Input, clamp(uv - offset, minCoord, maxCoord)));
    return vec4(pixel, 1.0);
}

void main(void)
{
    // base pixel colour for image
    vec4 pixel = vec4(1.0, 0.0, 1.0, 1.0);
    // get index in global work group i.e x,y position
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
  
    float u =1.0f * (gl_GlobalInvocationID.x + 0.5f)/gl_NumWorkGroups.x;
    float v =1.0f * (gl_GlobalInvocationID.y + 0.5f)/gl_NumWorkGroups.y;

    vec4 col = gaussianBlur(vec2(u,v));
    col = encodeRGBM(col.rgb, kRGBMRange);
    pixel= col;
    
    // output to a specific pixel in the image
    imageStore(img_output, pixel_coords, pixel);
}
