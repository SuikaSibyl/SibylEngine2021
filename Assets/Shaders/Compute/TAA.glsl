//////////////////////////////////////////////////////////////////////
/////                       Compute Shader                      //////
//////////////////////////////////////////////////////////////////////
#version 450 core
layout (local_size_x = 16, local_size_y = 16) in;
layout(rgba8, binding = 0) uniform image2D img_output;

layout(std430, binding=0) buffer Input
{
    vec2 OutputSize;
    float Alpha;
};

layout(binding = 0) uniform sampler2D u_PreviousFrame;
layout(binding = 1) uniform sampler2D u_CurrentFrame;
layout(binding = 2) uniform sampler2D u_Offset;

vec3 RGB2YCoCg(vec3 rgb)
{
    vec3 YCoCg;

	YCoCg.y = rgb.r - rgb.b;
	float temp = rgb.b + YCoCg.y / 2;
	YCoCg.z = rgb.g - temp;
	YCoCg.x = temp + YCoCg.z / 2;

    return YCoCg;
}

vec3 YCoCg2RGB(vec3 YCoCg)
{
    vec3 rgb;

	float temp = YCoCg.x - YCoCg.z / 2;
	rgb.g = YCoCg.z + temp;
	rgb.b = temp - YCoCg.y / 2;
	rgb.r = rgb.b + YCoCg.y;

    return rgb;
}

vec3 ClipAABB(vec3 aabbMin, vec3 aabbMax, vec3 prevSample)
{
	// note: only clips towards aabb center (but fast!)
	vec3 p_clip = 0.5 * (aabbMax + aabbMin);
	vec3 e_clip = 0.5 * (aabbMax - aabbMin);

	vec3 v_clip = prevSample - p_clip;
	vec3 v_unit = v_clip.xyz / e_clip;
	vec3 a_unit = abs(v_unit);
	float ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));

	if (ma_unit > 1.0)
		return p_clip + v_clip / ma_unit;
	else
		return prevSample;// point inside aabb
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

    vec2 Offset = texture2D(u_Offset, vec2(u,v)).xy;
    vec4 previousCol = texture2D(u_PreviousFrame, vec2(u,v) - Offset);
    vec3 previousColYcc = RGB2YCoCg(previousCol.rgb);
    
    vec4 currentCol = texture2D(u_CurrentFrame, vec2(u,v));

    vec3 currentCol_z0p1 = RGB2YCoCg(texture2D(u_CurrentFrame, vec2(u,v) + vec2(0, +dv)).xyz);
    vec3 currentCol_z0m1 = RGB2YCoCg(texture2D(u_CurrentFrame, vec2(u,v) + vec2(0, -dv)).xyz);
    vec3 currentCol_m1z0 = RGB2YCoCg(texture2D(u_CurrentFrame, vec2(u,v) + vec2(-dv, 0)).xyz);
    vec3 currentCol_p1z0 = RGB2YCoCg(texture2D(u_CurrentFrame, vec2(u,v) + vec2(+dv, 0)).xyz);
    
    vec3 min_currCol = min(min(currentCol_z0p1,currentCol_z0m1),min(currentCol_m1z0,currentCol_p1z0));
    vec3 max_currCol = max(max(currentCol_z0p1,currentCol_z0m1),max(currentCol_m1z0,currentCol_p1z0));
    vec3 cloped_prevCol = ClipAABB(min_currCol,max_currCol,previousColYcc);
    previousCol.rgb = YCoCg2RGB(cloped_prevCol);

    float unbiased_diff = abs(previousColYcc.x - cloped_prevCol.x) / max(previousColYcc.x, max(cloped_prevCol.x, 0.2));
	float unbiased_weight = 1.0 - unbiased_diff;
	float unbiased_weight_sqr = unbiased_weight * unbiased_weight;
    
    pixel.rgb= previousCol.rgb * unbiased_weight_sqr + currentCol.rgb * (1-unbiased_weight_sqr);
    pixel.rgb= previousCol.rgb * 0.95 + currentCol.rgb * 0.05;

    // output to a specific pixel in the image
    imageStore(img_output, pixel_coords, pixel);
}
