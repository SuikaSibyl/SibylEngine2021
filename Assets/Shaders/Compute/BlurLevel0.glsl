//////////////////////////////////////////////////////////////////////
/////                       Compute Shader                      //////
//////////////////////////////////////////////////////////////////////
#version 450 core
layout (local_size_x = 16, local_size_y = 16) in;
layout(rgba8, binding = 0) uniform image2D img_output;

layout(std430, binding=0) buffer Input
{
    vec2 OutputSize;
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

vec4 AddWeighted(vec2 uv, vec2 offset ,vec2 min, vec2 max, float weight)
{
    // vec2 uv_offseted = uv + offset;
    // if((uv_offseted.x > min.x) && (uv_offseted.y > min.y))
    // {
    //     return vec4(DecodeRGBM(texture(u_Input, uv_offseted)), weight);
    // }

    return vec4(0);
}

vec4 gaussianBlur(vec2 uv) {
    vec2 maxCoord = 1.0 - 1.0 / uTextureBlurInputSize.xy;
    vec2 minCoord = 1.0 / uTextureBlurInputSize.xy;

    vec3 pixel= 0.375 * DecodeRGBM(texture(u_Input, max(min(uv, maxCoord), minCoord)));
    vec2 offset;
    vec2 blurDir = uBlurDir.xy / OutputSize.xy;
    blurDir *= uGlobalTexSize.y * 0.00075;
    offset = blurDir * 1.2;

    pixel += 0.3125 *  DecodeRGBM(texture(u_Input, max(min(uv + offset, maxCoord), minCoord)));
    pixel += 0.3125 *  DecodeRGBM(texture(u_Input, max(min(uv - offset, maxCoord), minCoord)));
    vec2 offsetuv = uv - offset;
    return vec4(pixel, 1.0);
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

    vec4 col = gaussianBlur(vec2(u,v));
    col = encodeRGBM(col.rgb, kRGBMRange);
    pixel= col;
    
    // output to a specific pixel in the image
    imageStore(img_output, pixel_coords, pixel);
}
