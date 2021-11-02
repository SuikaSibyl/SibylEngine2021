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
    v_currPos = View * Model * vec4(aPos, 1.0);
}

//////////////////////////////////////////////////////////////////////
/////                     Fragment Shader                       //////
//////////////////////////////////////////////////////////////////////
#type FS
#version 330 core
// Vertex Inputs
in vec4 v_currPos;
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

void main()
{
    float alpha = 1.0;
    float depth = (v_currPos.z - ZNearFar.x) / (ZNearFar.y - ZNearFar.x);
    float scatter = 0.0;
    float profile = 0.0;

    // FragColor = vec4(depth,0,0,1);
    FragColor = encodeDepthAlphaProfileScatter(depth, alpha, profile, scatter);
}