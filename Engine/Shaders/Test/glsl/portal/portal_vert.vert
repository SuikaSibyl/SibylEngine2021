#version 450

layout(push_constant) uniform PushConstantObject {
    mat4 model;
} PushConstants;

layout(binding = 0) uniform PerViewUniformBuffer {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
} view_ubo;

struct Particle
{
    vec4 pos;
    vec4 vel;
    vec4 color;
    vec4 ext;
};
layout(set = 0, binding = 2, std430) buffer Particles
{
    Particle particle[];
} particles;

layout(binding = 3) uniform sampler2D texSampler;

layout(set = 0, binding = 4, std430) buffer LiveIndexBuffer
{
    uint[] indices;
} livePool;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;

layout (location = 0) out PerVertexData
{
  vec3 fragColor;
  vec2 fragTexCoord;
} v_out;   // [max_vertices]

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

float speed_y_curve(float i)
{
    if(i > 1) return 0.769;
    return 0.769 * i;
}

vec3 getScale(mat4 matrix)
{
    return vec3(1,1,1);
}

void main() {
    Particle particle = particles.particle[livePool.indices[gl_InstanceIndex.x]];
    vec3 instance_pos = particle.pos.xyz;

    mat4 billboardMat = billboardAlongVelocity(particle.vel.xyz, view_ubo.cameraPos.xyz - particle.pos.xyz);

    float clamped_speed = clamp(length(particle.vel.xyz), 0, 4) / 4;
    vec4 modelPosition = billboardMat * vec4(inPosition * vec3(0.2,0.2,0.2) * getScale(PushConstants.model) * vec3(0.1, speed_y_curve(clamped_speed), 1),1.0);

    gl_Position = view_ubo.proj * view_ubo.view * (vec4(modelPosition.xyz + instance_pos, 1.0));

    float lifeAlpha = 1 - particle.pos.w / particle.ext.x;
    vec4 colorOverLife = texture(texSampler, vec2((0.5 * (lifeAlpha) * 127)/128, 0.75));
    float intensityOverLife = texture(texSampler, vec2((0.5 * (lifeAlpha) * 127)/128, 0.25)).b;
    v_out.fragColor = particle.color.rgb *pow(2,intensityOverLife) * colorOverLife.rgb * colorOverLife.a;
    v_out.fragTexCoord = inTexCoord;
}