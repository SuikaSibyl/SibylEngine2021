#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec4 inTangent;

layout(location = 0) out vec2 fragUV0;

layout(push_constant) uniform PushConstantObject {
    mat4 model;
} PushConstants;

layout(binding = 0) uniform PerViewUniformBuffer {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
} view_ubo;

void main() {
    gl_Position = view_ubo.proj * view_ubo.view * PushConstants.model * vec4(inPosition, 1.0);
    gl_Position.z = (gl_Position.z + gl_Position.w) / 2.0;
    fragUV0 = inTexCoord;
}
