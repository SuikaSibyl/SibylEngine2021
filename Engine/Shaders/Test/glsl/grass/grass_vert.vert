#version 450

layout(push_constant) uniform PushConstantObject {
    mat4 model;
} PushConstants;

layout(binding = 0) uniform PerViewUniformBuffer {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
} view_ubo;

layout(set = 0, binding = 1, std430) buffer ParticlesPos
{ vec4 particle_pos[]; };

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;

mat4 billboardTowardCameraPlane(vec3 cameraPos)
{
    vec3 look = - normalize(cameraPos);
    vec3 right = normalize(cross(vec3(0,1,0), look));
    vec3 up = normalize(cross(right, look));
    return mat4(
        right.x,right.y,right.z,0,
        up.x,up.y,up.z,0,
        look.x,look.y,look.z,0,
        0,0,0,1);
}

vec3 getScale(mat4 matrix)
{
    return vec3(1,1,1);
}

layout (location = 0) out PerVertexData
{
  vec3 fragColor;
  vec2 fragTexCoord;
} v_out;

mat4 billboardAlongVelocity(vec3 velocity, vec3 cameraPos)
{
    vec3 up = - normalize(velocity);
    vec3 right = normalize(cross(up, -normalize(cameraPos)));
    vec3 look = normalize(cross(right, up));
    return mat4(
        right.x,right.y,right.z,0,
        up.x,up.y,up.z,0,
        look.x,look.y,look.z,0,
        0,0,0,1);
}

void main() {
    vec4 pos = particle_pos[gl_InstanceIndex.x];
    mat4 billboardMat = billboardAlongVelocity(vec3(0,1,0), view_ubo.cameraPos.xyz - pos.xyz);
    
    vec4 modelPosition = billboardMat * vec4(inPosition * vec3(5,5,5) * getScale(PushConstants.model),1.0);
    modelPosition.rgb += PushConstants.model[3].rgb;
    gl_Position = view_ubo.proj * view_ubo.view * (vec4(modelPosition.xyz + pos.xyz, 1.0));

    vec2 uv = inTexCoord;
    uv.x = (pos.w + uv.x) * 0.25;
    v_out.fragTexCoord = uv;
}