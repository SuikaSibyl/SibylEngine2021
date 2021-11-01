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
    vec3 pixel = 0.2255859375 * DecodeRGBM(texture2D(u_Input, uv));
    vec2 offset;
    vec2 blurDir = uBlurDir.xy / gl_NumWorkGroups.xy;
    blurDir *= uGlobalTexSize.y * 0.00075;
    offset = blurDir * 1.3846153846153846;
    pixel += 0.314208984375 * DecodeRGBM(texture(u_Input, clamp(uv + offset, vec2(0,0), vec2(1,1))));
    pixel += 0.314208984375 * DecodeRGBM(texture(u_Input, clamp(uv - offset, vec2(0,0), vec2(1,1))));
    offset = blurDir * 3.230769230769231;
    pixel += 0.06982421875 * DecodeRGBM(texture(u_Input, clamp(uv + offset, vec2(0,0), vec2(1,1))));
    pixel += 0.06982421875 * DecodeRGBM(texture(u_Input, clamp(uv - offset, vec2(0,0), vec2(1,1))));
    offset = blurDir * 5.076923076923077;
    pixel += 0.003173828125 * DecodeRGBM(texture(u_Input, clamp(uv + offset, vec2(0,0), vec2(1,1))));
    pixel += 0.003173828125 * DecodeRGBM(texture(u_Input, clamp(uv - offset, vec2(0,0), vec2(1,1))));
    return vec4(pixel, 1.0);

    vec3 pixel = 0.20947265625 *  (vec4(decodeRGBM(texture2D(TextureBlurInput, (min(gTexCoord.xy, 1.0 - 1e+0 / uTextureBlurInputSize.xy)) * uTextureBlurInputRatio), uRGBMRange), 1.0)).rgb;
    vec2 offset;
    vec2 blurDir = uPixelRatio.xy * uBlurDir.xy / gl_NumWorkGroups.xy;
    blurDir *= uGlobalTexSize.y * 0.00075;
    offset = blurDir * 1.4;
    pixel += 0.30548095703125 *  (vec4(decodeRGBM(texture2D(TextureBlurInput, (min(gTexCoord.xy + offset.xy, 1.0 - 1e+0 / uTextureBlurInputSize.xy)) * uTextureBlurInputRatio), uRGBMRange), 1.0)).rgb;
    pixel += 0.30548095703125 *  (vec4(decodeRGBM(texture2D(TextureBlurInput, (min(gTexCoord.xy - offset.xy, 1.0 - 1e+0 / uTextureBlurInputSize.xy)) * uTextureBlurInputRatio), uRGBMRange), 1.0)).rgb;
    offset = blurDir * 3.2666666666666666;
    pixel += 0.08331298828125 *  (vec4(decodeRGBM(texture2D(TextureBlurInput, (min(gTexCoord.xy + offset.xy, 1.0 - 1e+0 / uTextureBlurInputSize.xy)) * uTextureBlurInputRatio), uRGBMRange), 1.0)).rgb;
    pixel += 0.08331298828125 *  (vec4(decodeRGBM(texture2D(TextureBlurInput, (min(gTexCoord.xy - offset.xy, 1.0 - 1e+0 / uTextureBlurInputSize.xy)) * uTextureBlurInputRatio), uRGBMRange), 1.0)).rgb;
    offset = blurDir * 5.133333333333334;
    pixel += 0.00640869140625 *  (vec4(decodeRGBM(texture2D(TextureBlurInput, (min(gTexCoord.xy + offset.xy, 1.0 - 1e+0 / uTextureBlurInputSize.xy)) * uTextureBlurInputRatio), uRGBMRange), 1.0)).rgb;
    pixel += 0.00640869140625 *  (vec4(decodeRGBM(texture2D(TextureBlurInput, (min(gTexCoord.xy - offset.xy, 1.0 - 1e+0 / uTextureBlurInputSize.xy)) * uTextureBlurInputRatio), uRGBMRange), 1.0)).rgb;
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
