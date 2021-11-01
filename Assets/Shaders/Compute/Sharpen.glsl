//////////////////////////////////////////////////////////////////////
/////                       Compute Shader                      //////
//////////////////////////////////////////////////////////////////////
#version 450 core
layout (local_size_x = 1, local_size_y = 1) in;
layout(rgba8, binding = 0) uniform image2D img_output;

layout(std430, binding=0) buffer Input
{
    float uSharpFactor;
};

layout(binding = 0) uniform sampler2D u_Texture;
layout(binding = 0) uniform sampler2D u_Depth;

vec3 sharpColorFactor(const in vec3 color, const float sharp, const in vec2 uv) {
    vec2 maxCoord = vec2(1.0) - vec2(1.0) / gl_NumWorkGroups.xy;
    vec2 minCoord = vec2(1.0) / gl_NumWorkGroups.xy;
    vec2 off = minCoord;

    vec3 rgbNW = (texture(u_Texture, clamp(uv + off * vec2(-1.0, -1.0), minCoord, maxCoord))).rgb;
    vec3 rgbSE = (texture(u_Texture, clamp(uv + off * vec2(1.0, 1.0), minCoord, maxCoord))).rgb;
    vec3 rgbNE = (texture(u_Texture, clamp(uv + off * vec2(1.0, -1.0), minCoord, maxCoord))).rgb;
    vec3 rgbSW = (texture(u_Texture, clamp(uv + off * vec2(-1.0, 1.0), minCoord, maxCoord))).rgb;
    return color + sharp * (4.0 * color - rgbNW - rgbNE - rgbSW - rgbSE);
}

vec4 sharpen(const in vec4 color, const in vec2 uv) {
    float alpha = (texture(u_Depth, uv)).a;
    if( alpha == 0.0 ) {
        return vec4(color.rgb, 1.0);
    }
    return vec4(sharpColorFactor(color.rgb, uSharpFactor * alpha, uv), color.a);
}

vec4 sharpen(const in vec2 uv) {
    return sharpen( (texture(u_Texture, uv)), uv);
}

void main(void)
{
    // base pixel colour for image
    vec4 pixel = vec4(1.0, 0.0, 1.0, 1.0);
    // get index in global work group i.e x,y position
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
  
    float u =1.0f * (gl_GlobalInvocationID.x + 0.5f)/gl_NumWorkGroups.x;
    float v =1.0f * (gl_GlobalInvocationID.y + 0.5f)/gl_NumWorkGroups.y;

    vec4 fxaaCol = sharpen(vec2(u,v));
    pixel= vec4(fxaaCol.rgb, 1);

    // output to a specific pixel in the image
    imageStore(img_output, pixel_coords, pixel);
}
