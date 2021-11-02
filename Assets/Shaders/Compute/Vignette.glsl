//////////////////////////////////////////////////////////////////////
/////                       Compute Shader                      //////
//////////////////////////////////////////////////////////////////////
#version 450 core
layout (local_size_x = 16, local_size_y = 16) in;
layout(rgba8, binding = 0) uniform image2D img_output;

layout(std430, binding=0) buffer Input
{
    vec2 OutputSize;
    vec2 uLensRadius;
    float uFrameMod;
};

uniform sampler2D u_Texture;

vec3 linearTosRGB(const in vec3 color) {
    return vec3( color.r < 0.0031308 ? color.r * 12.92 : 1.055 * pow(color.r, 1.0/2.4) - 0.055, color.g < 0.0031308 ? color.g * 12.92 : 1.055 * pow(color.g, 1.0/2.4) - 0.055, color.b < 0.0031308 ? color.b * 12.92 : 1.055 * pow(color.b, 1.0/2.4) - 0.055);
}

vec3 sRGBToLinear(const in vec3 color) {
    return vec3( color.r < 0.04045 ? color.r * (1.0 / 12.92) : pow((color.r + 0.055) * (1.0 / 1.055), 2.4), color.g < 0.04045 ? color.g * (1.0 / 12.92) : pow((color.g + 0.055) * (1.0 / 1.055), 2.4), color.b < 0.04045 ? color.b * (1.0 / 12.92) : pow((color.b + 0.055) * (1.0 / 1.055), 2.4));
}

float interleavedGradientNoise(const in vec2 fragCoord, const in float frameMod) {
    vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(fragCoord.xy + frameMod * vec2(47.0, 17.0) * 0.695, magic.xy)));
}

float vignetteDithered(const in vec2 uv) {
    vec2 lens = uLensRadius;
    lens.y = min(lens.y, lens.x - 1e-4);
    float jitter = interleavedGradientNoise(gl_GlobalInvocationID.xy, uFrameMod);
    jitter = (lens.x - lens.y) * (lens.x + lens.y) * 0.07 * (jitter - 0.5);
    return smoothstep(lens.x, lens.y, jitter + distance(uv, vec2(0.5)));
}
vec4 vignette(const in vec4 color, const in vec2 uv) {
    float factor = vignetteDithered(uv);
    return vec4(linearTosRGB(sRGBToLinear(color.rgb) * factor), clamp(color.a + (1.0 - factor), 0.0, 1.0));
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
    vec4 vignetteCol = vignette(textCol, vec2(u,v));
    pixel= vignetteCol;

    // output to a specific pixel in the image
    imageStore(img_output, pixel_coords, pixel);
}
