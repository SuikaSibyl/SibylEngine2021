//////////////////////////////////////////////////////////////////////
/////                       Compute Shader                      //////
//////////////////////////////////////////////////////////////////////
#version 450 core
layout (local_size_x = 1, local_size_y = 64) in;

layout(rgba8, binding = 0) uniform image2D img_output;

layout(std430, binding=0) buffer Input
{
    vec2 OutputSize;
    float u_radius;
};

layout(binding = 0) uniform sampler2D u_Input;

#define MAXRADIUS 5
shared vec3 imagePixel[gl_WorkGroupSize.y + 2*MAXRADIUS];

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

    int radius = int(u_radius);

    if(gl_LocalInvocationID.y < radius)
    {
        float extrav = dv * (max(gl_GlobalInvocationID.y - radius, 0) + 0.5f);
        imagePixel[gl_LocalInvocationID.y] = texture(u_Input, vec2(u,extrav)).rgb;
    }

    if(gl_LocalInvocationID.y >= gl_WorkGroupSize.y - radius)
    {
        float extrav = dv * (min(gl_GlobalInvocationID.y + radius, OutputSize.y - 1) + 0.5f);
        imagePixel[gl_LocalInvocationID.y + 2 * radius] = texture(u_Input, vec2(u,extrav)).rgb;
    }

    imagePixel[gl_LocalInvocationID.y + radius] = texture(u_Input, vec2(u,v)).rgb;
    memoryBarrierShared();
    barrier();

    vec3 blurColor = vec3(0);
    float weight = 1 * 1. / (2 * radius + 1);
    for(int i=-radius; i<=radius;i++)
    {
        int k = i + int(gl_LocalInvocationID.y) + radius;
        blurColor += weight * imagePixel[k];
    }

    pixel.rgb= blurColor;

    // output to a specific pixel in the image
    imageStore(img_output, pixel_coords, pixel);
}
