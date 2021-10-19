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

struct StandardForwardV2F
{
    vec3 tspace0;
    vec3 tspace1;
    vec3 tspace2;
};
out StandardForwardV2F v2f;

// Vertex Outputs
out vec3 v_Color;
out vec2 v_TexCoord;
out vec4 v_currPos;
out vec4 v_prevPos;
// out vec3 v_normal;

// Uniform Constants
uniform mat4 Model;
uniform mat4 View;
uniform mat4 Projection;
uniform mat4 PreviousPV;
uniform mat4 CurrentPV;
uniform vec4 Color;

void RegularVertexInit()
{
    mat3 normalMatrix = mat3(transpose(inverse(Model)));
    // Clac TBN
    vec3 wNormal = normalMatrix * normalize(aNormal);
    vec3 wTangent = mat3(Model) * (aTangent.xyz);
    vec3 tangentSign = vec3(aTangent.w);
    // Cross({World Space} Normal, Tangent), Get Bitangent
    vec3 wBitangent = cross(wNormal, wTangent) * tangentSign;
    v2f.tspace0 = vec3(wTangent.x, wBitangent.x, wNormal.x);
    v2f.tspace1 = vec3(wTangent.y, wBitangent.y, wNormal.y);
    v2f.tspace2 = vec3(wTangent.z, wBitangent.z, wNormal.z);
}

void main()
{
    gl_Position = Projection * View * Model * vec4(aPos, 1.0);
    v_currPos = CurrentPV * Model * vec4(aPos, 1.0);
    v_prevPos = PreviousPV * Model * vec4(aPos, 1.0);

    RegularVertexInit();

    // v_normal = mat3(transpose(inverse(Model))) * normalize(aNormal);

    v_Color = aPos;
    v_TexCoord = aUV;
}

//////////////////////////////////////////////////////////////////////
/////                     Fragment Shader                       //////
//////////////////////////////////////////////////////////////////////
#type FS
#version 330 core

struct StandardForwardV2F
{
    vec3 tspace0;
    vec3 tspace1;
    vec3 tspace2;
};
in StandardForwardV2F v2f;

// Vertex Inputs
in vec3 v_Color;
in vec2 v_TexCoord;
in vec4 v_currPos;
in vec4 v_prevPos;
// in vec3 v_normal;

// Fragment outputs
layout(location = 0) out vec4 FragColor;  
layout(location = 1) out vec4 UVOffset;

// Uniform items
uniform vec4 Color;
uniform sampler2D u_Texture;

// ==============================
// Point Lights
// ==============================
uniform int DirectionalLightNum;
uniform int PointLightNum;

struct DirectionalLight {
    vec3 direction;
    float intensity;
    vec3 color;
};  
#define DIRECTIONAL_LIGHTS_MAX 4
uniform DirectionalLight directionalLights[DIRECTIONAL_LIGHTS_MAX];

// ==============================
// Point Lights
// ==============================
struct PointLight {
    vec3 position;
    float intensity;
    vec3 color;
};  
#define POINT_LIGHTS_MAX 4
uniform PointLight pointLights[POINT_LIGHTS_MAX];

struct Light {
    vec3 direction;
    float intensity;
    vec3 color;
};

Light[DIRECTIONAL_LIGHTS_MAX+POINT_LIGHTS_MAX] PrepareLights()
{
    Light lights[DIRECTIONAL_LIGHTS_MAX+POINT_LIGHTS_MAX];
    for(int i=0;i<DirectionalLightNum;i++)
    {
        lights[i].direction=directionalLights[i].direction;
        lights[i].intensity=directionalLights[i].intensity;
        lights[i].color=directionalLights[i].color;
    }
    return lights;
}

vec3 NormalTangentToWorld(vec3 tNormal)
{
    vec3 wNormal;
    wNormal.x = dot(v2f.tspace0, tNormal);
    wNormal.y = dot(v2f.tspace1, tNormal);
    wNormal.z = dot(v2f.tspace2, tNormal);
    return normalize(wNormal);
}

vec3 GetWorldNormal()
{
    // vec4 nortex = tex2D(_NormalTex, i.uv);
    vec3 tNormal = normalize(vec3(0,0,1));
    vec3 wNormal = NormalTangentToWorld(tNormal);
    return gl_FrontFacing ? -wNormal : wNormal;
}

void main()
{
    FragColor = vec4(0,0,0,1);

    vec4 tex = texture(u_Texture, v_TexCoord);
    if(tex.a < 0.1)
        discard;

    vec3 normal = GetWorldNormal();

    Light[DIRECTIONAL_LIGHTS_MAX+POINT_LIGHTS_MAX] Lights = PrepareLights();
    for(int i=0;i<DirectionalLightNum;i++)
    {
        float cos = dot(normal, -Lights[0].direction);
        if(cos<0) cos=0;
        FragColor.xyz += Color.xyz * tex.xyz * cos;
    }

    vec2 offset = v_currPos.xy/v_currPos.w - v_prevPos.xy/v_prevPos.w;
    UVOffset = vec4(offset * 0.5,1.0,1.0);
}