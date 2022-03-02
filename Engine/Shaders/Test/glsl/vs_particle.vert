#version 450

layout(binding = 0) uniform UniformBufferObject {
    vec4 cameraPos;
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(set = 0, binding = 2, std430) buffer Positions
{
    vec4 pos[];
} positions;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;

void main() {
    vec3 instance_pos = positions.pos[gl_InstanceIndex.x].xyz;

    vec3 look = normalize(ubo.cameraPos.xyz);
    vec3 right = normalize(cross(vec3(0,1,0), look));
    vec3 up = normalize(cross(look, right));
    mat4 billboardMat = mat4(
        right.x,right.y,right.z,0,
        up.x,up.y,up.z,0,
        look.x,look.y,look.z,0,
        0,0,0,1);

    vec4 modelPosition = ubo.model * vec4(inPosition,1.0);

    gl_Position = ubo.proj * ubo.view * billboardMat * (vec4(modelPosition.xyz + instance_pos, 1.0));
    fragColor = inColor;
    fragTexCoord = inTexCoord;
}