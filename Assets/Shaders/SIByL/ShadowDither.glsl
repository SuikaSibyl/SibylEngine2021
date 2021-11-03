//////////////////////////////////////////////////////////////////////
/////                       Vertex Shader                       //////
//////////////////////////////////////////////////////////////////////
#type VS
#version 330 core

// VData Inputs
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec4 aTangent;
layout (location = 3) in vec2 aUV;

// Vertex Outputs
out vec4 v_currPos;
out vec2 v_TexCoord;

// Uniform Constants
uniform mat4 Model;
uniform mat4 View;
uniform mat4 Projection;
uniform mat4 ProjectionDither;
uniform mat4 PreviousPV;
uniform mat4 CurrentPV;
uniform vec4 Color;
uniform vec4 ZNearFar;

void main()
{
    gl_Position = ProjectionDither * View * Model * vec4(aPos, 1.0);
    v_currPos = gl_Position;
    v_TexCoord = aUV;
}

//////////////////////////////////////////////////////////////////////
/////                     Fragment Shader                       //////
//////////////////////////////////////////////////////////////////////
#type FS
#version 330 core
// Vertex Inputs
in vec4 v_currPos;
in vec2 v_TexCoord;

// Fragment outputs
layout(location = 0) out vec4 FragColor;  
// Uniform items
uniform vec4 Color;
uniform sampler2D u_Main;
uniform vec4 ZNearFar;

vec4 encodeDepthAlphaProfileScatter(const in float depth, const in float alpha, const in float profile, const in float scatter) {
    vec4 pack = vec4(0.0);
    pack.a = alpha;
    if(profile == 0.0) {
        const vec3 code = vec3(1.0, 255.0, 65025.0);
        pack.rgb = vec3(code * depth);
        pack.gb = fract(pack.gb);
        pack.rg -= pack.gb * (1.0 / 256.0);
    }
    else {
        pack.g = fract(depth * 255.0);
        pack.r = depth - pack.g / 255.0;
        pack.b = floor(0.5 + scatter * 63.0) * 4.0 / 255.0;
    }
    pack.b -= mod(pack.b, 4.0 / 255.0);
    pack.b += profile / 255.0;
    return pack;
}

float getMaterialOpacity() {
    float alpha = 1.0;
    alpha = (texture(u_Main, v_TexCoord).a);
    return alpha;
}

float pseudoRandom(const in vec2 fragCoord) {
    vec3 p3 = fract(vec3(fragCoord.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}

void ditheringMaskingDiscard(
const in vec4 fragCoord, const in int dithering, const in float alpha, const in float factor, const in vec2 halton) {
    float rnd;
    rnd = pseudoRandom(fragCoord.xy + halton * 1000.0 + fragCoord.z * 1000.0);
    if ((alpha * factor) < rnd) discard;
}

float decode24(const in vec3 x) {
    const vec3 decode = 1.0 / vec3(1.0, 255.0, 65025.0);
    return dot(x, decode);
}

void main()
{
    ditheringMaskingDiscard(gl_FragCoord, 1, getMaterialOpacity(), 1, ZNearFar.zw);
    float alpha = 1.0;
    float depth = (v_currPos.z + 1) / 2;
    float scatter = 0.0;
    float profile = 0.0;

    FragColor = encodeDepthAlphaProfileScatter(depth, alpha, profile, scatter);
}