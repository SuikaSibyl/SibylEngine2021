//////////////////////////////////////////////////////////////////////
/////                       Vertex Shader                       //////
//////////////////////////////////////////////////////////////////////
#type VS
#version 330 core

#define DIRECTIONAL_LIGHTS_MAX 4

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
out vec4 v_WSCurrPos;
out vec4 v_currPos;
out vec4 v_prevPos;
out vec3 v_vNormal;
out vec3 v_vLSPos[DIRECTIONAL_LIGHTS_MAX];
// out vec3 v_normal;

// Uniform Constants
uniform mat4 Model;

uniform mat4 View;
uniform mat4 Projection;
uniform mat4 ProjectionDither;
uniform mat4 PreviousPV;
uniform mat4 CurrentPV;
uniform vec4 ViewPos;
uniform vec4 ZNearFar;


uniform vec4 Color;

struct DirectionalLight {
    mat4 projview;
    vec3 direction;
    float intensity;
    vec3 color;
};  
uniform DirectionalLight directionalLights[DIRECTIONAL_LIGHTS_MAX];


void main()
{
    gl_Position = ProjectionDither * View * Model * vec4(aPos, 1.0);
    v_prevPos = PreviousPV * Model * vec4(aPos, 1.0);

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

    v_WSCurrPos = Model * vec4(aPos, 1.0);
    v_currPos = CurrentPV * v_WSCurrPos;
    v_Color = aPos;
    v_TexCoord = aUV;
    v_vNormal = mat3(transpose(inverse(View * Model))) * normalize(aNormal);

    for(int i =0; i<DIRECTIONAL_LIGHTS_MAX; i++)
    {
        v_vLSPos[i] = (directionalLights[i].projview * v_WSCurrPos).rgb;
    }
}

//////////////////////////////////////////////////////////////////////
/////                     Fragment Shader                       //////
//////////////////////////////////////////////////////////////////////
#type FS
#version 330 core

#define DIRECTIONAL_LIGHTS_MAX 4

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
in vec4 v_WSCurrPos;
in vec4 v_currPos;
in vec4 v_prevPos;
in vec3 v_vNormal;
in vec3 v_vLSPos[DIRECTIONAL_LIGHTS_MAX];

// in vec3 v_normal;

// Fragment outputs
layout(location = 0) out vec4 FragColor;  
layout(location = 1) out vec4 UVOffset; 
layout(location = 2) out vec4 Normal;


// Uniform items
uniform vec4 Color;
uniform sampler2D u_Main;
uniform sampler2D u_Normal;
uniform sampler2D u_DirectionalShadowmap;

// ==============================
// Point Lights
// ==============================
uniform vec4 ViewPos;

uniform int DirectionalLightNum;
uniform int PointLightNum;

struct DirectionalLight {
    mat4 projview;
    vec3 direction;
    float intensity;
    vec3 color;
};  

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
    mat4 projview;
    vec3 direction;
    float intensity;
    vec3 color;
};

Light[DIRECTIONAL_LIGHTS_MAX+POINT_LIGHTS_MAX] PrepareLights()
{
    Light lights[DIRECTIONAL_LIGHTS_MAX+POINT_LIGHTS_MAX];
    for(int i=0;i<DirectionalLightNum;i++)
    {
        lights[i].projview=directionalLights[i].projview;
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
    vec4 nortex = texture(u_Normal, v_TexCoord);
    vec3 tNormal = normalize(nortex.xyz * 2 - vec3(1,1,1));
    vec3 wNormal = NormalTangentToWorld(tNormal);
    return gl_FrontFacing ? -wNormal : wNormal;
}

const float kRGBMRange = 2.0;
vec4 EncodeRGBM(vec3 color)
{
    color *= 1.0 / kRGBMRange;
    float m = max(max(color.x, color.y), max(color.z, 1e-5));
    m = ceil(m * 255) / 255;
    return vec4(color / m, m);
}

vec4 precomputeGGX(const in vec3 normal, const in vec3 eyeVector, const in float roughness) {
    float NoV = clamp(dot(normal, eyeVector), 0., 1.);
    float r2 = roughness * roughness;
    return vec4(r2, r2 * r2, NoV, NoV * (1.0 - r2));
}
float D_GGX(const vec4 precomputeGGX, const float NoH) {
    float a2 = precomputeGGX.y;
    float d = (NoH * a2 - NoH) * NoH + 1.0;
    return a2 / (3.141593 * d * d);
}
vec3 F_Schlick(const vec3 f0, const float f90, const in float VoH) {
    float VoH5 = pow(1.0 - VoH, 5.0);
    return f90 * VoH5 + (1.0 - VoH5) * f0;
}
float F_Schlick(const float f0, const float f90, const in float VoH) {
    return f0 + (f90 - f0) * pow(1.0 - VoH, 5.0);
}
float V_SmithCorrelated(const vec4 precomputeGGX, const float NoL) {
    float a = precomputeGGX.x;
    float smithV = NoL * (precomputeGGX.w + a);
    float smithL = precomputeGGX.z * (NoL * (1.0 - a) + a);
    return 0.5 / (smithV + smithL);
}
vec3 specularLobe(const vec4 precomputeGGX, const vec3 normal, const vec3 eyeVector, const vec3 eyeLightDir, const vec3 specular, const float NoL, const float f90) {
    vec3 H = normalize(eyeVector + eyeLightDir);
    float NoH = clamp(dot(normal, H), 0., 1.);
    float VoH = clamp(dot(eyeLightDir, H), 0., 1.);
    float D = D_GGX(precomputeGGX, NoH);
    float V = V_SmithCorrelated(precomputeGGX, NoL);
    vec3 F = F_Schlick(specular, f90, VoH);
    return (D * V * 3.141593) * F;
}

float decode24(const in vec3 x) {
    const vec3 decode = 1.0 / vec3(1.0, 255.0, 65025.0);
    return dot(x, decode);
}

float GetDepth(const in vec2 LSuv, const in vec2 uvoffset,const in vec2 pcfoffset)
{
    vec2 uvspaceUV = (LSuv + vec2(1)) * vec2(0.5, 0.5);
    if(uvspaceUV.x > 1 || uvspaceUV.y > 1 || uvspaceUV.x < 0 || uvspaceUV.y < 0)
        return 99999;
    vec4 tmp = texture2D(u_DirectionalShadowmap, uvspaceUV * vec2(0.5) + uvoffset + pcfoffset);
    if(tmp.a==0) return 99999;
    else return decode24(tmp.rgb);
}

void computeLightLambertGGX(
const in vec3 normal, const in vec3 eyeVector, const in float NoL, const in vec4 precomputeGGX, const in vec3 diffuse, const in vec3 specular, const in float attenuation, const in vec3 lightColor, const in vec3 eyeLightDir, const in float f90, out vec3 diffuseOut, out vec3 specularOut, out bool lighted) {
    lighted = NoL > 0.0;
    if (lighted == false) {
        specularOut = diffuseOut = vec3(0.0);
        return;
    }
    vec3 colorAttenuate = attenuation * NoL * lightColor;
    specularOut = colorAttenuate * specularLobe(precomputeGGX, normal, eyeVector, eyeLightDir, specular, NoL, f90);
    diffuseOut = colorAttenuate * diffuse;
}

void main()
{
    FragColor = vec4(0,0,0,1);

    vec4 materialDiffuse = texture(u_Main, v_TexCoord);
    if(materialDiffuse.a < 0.1)
        discard;
    
    materialDiffuse.rgb *= Color.rgb;

    float materialRoughness = 0.5f;
    vec3 eyeVector = normalize(ViewPos.xyz - v_WSCurrPos.xyz);
    vec3 normal = GetWorldNormal();
    vec4 prepGGX = precomputeGGX(normal, eyeVector, max(0.045, materialRoughness));

    vec3 materialSpecular = vec3(0,0,0);
    float attenuation = 1;
    float materialF90 = clamp(50.0 * materialSpecular.g, 0.0, 1.0);

    vec3 lightDiffuse;
    vec3 lightSpecular;
    bool lighted;

    Light[DIRECTIONAL_LIGHTS_MAX+POINT_LIGHTS_MAX] Lights = PrepareLights();
    vec2 uvoffset[4];
    uvoffset[0] = vec2(0,0);
    uvoffset[1] = vec2(0.5,0);
    uvoffset[2] = vec2(0,0.5);
    uvoffset[3] = vec2(0.5,0.5);
    for(int i=0;i<DirectionalLightNum;i++)
    {
        float dotNL = dot(normal, -Lights[i].direction);
        if(dotNL<0) dotNL=0;

        // Shadow
        vec3 LSpos = v_vLSPos[i];

        float actualDistance = LSpos.z;
        float bias = max(0.05 * (1.0 - dotNL), 0.005);
        float shadow = 0.0;
        vec2 texelSize = 1.0 / vec2(1024);
        for(int x = -1; x <= 1; ++x)
        {
            for(int y = -1; y <= 1; ++y)
            {
                float lightDistance = GetDepth(LSpos.xy, uvoffset[i], vec2(x, y) * texelSize);
                lightDistance = lightDistance * 2 - 1;
                shadow += actualDistance - bias > lightDistance ? 0.0 : 1.0;        
            }    
        }
        shadow /= 9.0;

        float shadowIntensity = shadow;


        computeLightLambertGGX(normal, eyeVector, dotNL, prepGGX, materialDiffuse.rgb, 
            materialSpecular, attenuation, Lights[i].color, -Lights[i].direction, 
            materialF90, lightDiffuse, lightSpecular, lighted);
        FragColor.xyz += shadowIntensity * (lightDiffuse + lightSpecular);
    }

    FragColor = EncodeRGBM(FragColor.xyz);
    Normal = vec4(normalize(v_vNormal)*0.5 + vec3(0.5,0.5,0.5), 1);
    vec2 offset = v_currPos.xy/v_currPos.w - v_prevPos.xy/v_prevPos.w;
    UVOffset = vec4(offset * 0.5,1.0,1.0);
}