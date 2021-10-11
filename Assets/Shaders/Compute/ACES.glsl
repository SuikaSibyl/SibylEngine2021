//////////////////////////////////////////////////////////////////////
/////                       Compute Shader                      //////
//////////////////////////////////////////////////////////////////////
#version 450 core
layout (local_size_x = 1, local_size_y = 1) in;
layout(rgba8, binding = 0) uniform image2D img_output;
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

void main(void)
{
    // base pixel colour for image
    vec4 pixel = vec4(1.0, 0.0, 1.0, 1.0);
    // get index in global work group i.e x,y position
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
  
    float u =1.0f * (gl_GlobalInvocationID.x + 0.5f)/gl_NumWorkGroups.x;
    float v =1.0f * (gl_GlobalInvocationID.y + 0.5f)/gl_NumWorkGroups.y;

    vec4 textCol = texture(u_Texture, vec2(u,v));
    // textCol.xyz = ACESToneMapping(textCol.xyz,0.5);
    pixel= textCol;

    // output to a specific pixel in the image
    imageStore(img_output, pixel_coords, pixel);
}
