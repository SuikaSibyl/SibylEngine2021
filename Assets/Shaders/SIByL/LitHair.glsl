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

#define DIRECTIONAL_LIGHTS_MAX 4
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

struct DirectionalLight {
    mat4 projview;
    vec3 direction;
    float intensity;
    vec3 color;
};  
uniform DirectionalLight directionalLights[DIRECTIONAL_LIGHTS_MAX];


uniform vec4 Color;

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
uniform sampler2D u_DiffuseAO;
uniform sampler2D u_IBLLUT;
uniform sampler2D u_DirectionalShadowmap;

uniform samplerCube u_SpecularCube;
// ==============================
// Point Lights
// ==============================
uniform vec4 ViewPos;
uniform mat4 View;

uniform int DirectionalLightNum;
uniform int PointLightNum;
uniform vec4 ZNearFar;

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

float V_SmithGGXCorrelated_Anisotropic(float at, float ab, float ToV, float BoV, float ToL, float BoL, float NoV, float NoL) {
    float lambdaV = NoL * length(vec3(at * ToV, ab * BoV, NoV));
    float lambdaL = NoV * length(vec3(at * ToL, ab * BoL, NoL));
    return 0.5 / (lambdaV + lambdaL);
}
float D_GGX_Anisotropic(const float at, const float ab, const float ToH, const float BoH, const float NoH) {
    float a2 = at * ab;
    vec3 d = vec3(ab * ToH, at * BoH, a2 * NoH);
    float x = a2 / dot(d, d);
    return a2 * (x * x) / 3.141593;
}

vec3 anisotropicLobe(
const vec4 precomputeGGX, const vec3 normal, const vec3 eyeVector, const vec3 eyeLightDir, const vec3 specular, const float NoL, const float f90, const in vec3 anisotropicT, const in vec3 anisotropicB, const in float anisotropy) {
    vec3 H = normalize(eyeVector + eyeLightDir);
    float NoH = clamp(dot(normal, H), 0., 1.);
    float NoV = clamp(dot(normal, eyeVector), 0., 1.);
    float VoH = clamp(dot(eyeLightDir, H), 0., 1.);
    float ToV = dot(anisotropicT, eyeVector);
    float BoV = dot(anisotropicB, eyeVector);
    float ToL = dot(anisotropicT, eyeLightDir);
    float BoL = dot(anisotropicB, eyeLightDir);
    float ToH = dot(anisotropicT, H);
    float BoH = dot(anisotropicB, H);
    float aspect = sqrt(1.0 - abs(anisotropy) * 0.9);
    if (anisotropy > 0.0) aspect = 1.0 / aspect;
    float at = precomputeGGX.x * aspect;
    float ab = precomputeGGX.x / aspect;
    float D = D_GGX_Anisotropic(at, ab, ToH, BoH, NoH);
    float V = V_SmithGGXCorrelated_Anisotropic(at, ab, ToV, BoV, ToL, BoL, NoV, NoL);
    vec3 F = F_Schlick(specular, f90, VoH);
    return (D * V * 3.141593) * F;
}

void computeLightLambertGGXAnisotropy(
const in vec3 normal, const in vec3 eyeVector, const in float NoL, const in vec4 precomputeGGX, const in vec3 diffuse, const in vec3 specular, const in float attenuation, const in vec3 lightColor, const in vec3 eyeLightDir, const in float f90, const in vec3 anisotropicT, const in vec3 anisotropicB, const in float anisotropy, out vec3 diffuseOut, out vec3 specularOut, out bool lighted) {
    lighted = NoL > 0.0;
    if (lighted == false) {
        specularOut = diffuseOut = vec3(0.0);
        return;
    }
    vec3 colorAttenuate = attenuation * NoL * lightColor;
    specularOut = colorAttenuate * anisotropicLobe(precomputeGGX, normal, eyeVector, eyeLightDir, specular, NoL, f90, anisotropicT, anisotropicB, anisotropy);
    diffuseOut = colorAttenuate * diffuse;
}

float specularOcclusion(const in int occlude, const in float ao, const in vec3 normal, const in vec3 eyeVector) {
    if (occlude == 0) return 1.0;
    float d = dot(normal, eyeVector) + ao;
    return clamp((d * d) - 1.0 + ao, 0.0, 1.0);
}

const mat3 uEnvironmentTransform = mat3(
        0.9984, -0.0000, 0.0566, 
        0.0318, 0.8276, -0.5604, 
        -0.0469, 0.5613, 0.8263);

vec3 computeDiffuseSPH(const in vec3 normal) {
    vec3 uDiffuseSPH[9];
    uDiffuseSPH[0] = vec3(0.0575, 0.0620, 0.0618);
    uDiffuseSPH[1] = vec3(0.0061, 0.0082, 0.0090);
    uDiffuseSPH[2] = vec3(0.0129, 0.0150, 0.0171);
    uDiffuseSPH[3] = vec3(-0.0326, -0.0382, -0.0410);
    uDiffuseSPH[4] = vec3(-0.0085, -0.0102, -0.0129);
    uDiffuseSPH[5] = vec3(0.0046, 0.0055, 0.0075);
    uDiffuseSPH[6] = vec3(0.0047, 0.0054, 0.0055);
    uDiffuseSPH[7] = vec3(-0.0221, -0.0255, -0.0309);
    uDiffuseSPH[8] = vec3(0.0103, 0.0125, 0.0131);

    vec3 n = uEnvironmentTransform * normal;
    vec3 result = uDiffuseSPH[0] +
        uDiffuseSPH[1] * n.y +
        uDiffuseSPH[2] * n.z +
        uDiffuseSPH[3] * n.x +
        uDiffuseSPH[4] * n.y * n.x +
        uDiffuseSPH[5] * n.y * n.z +
        uDiffuseSPH[6] * (3.0 * n.z * n.z - 1.0) +
        uDiffuseSPH[7] * (n.z * n.x) +
        uDiffuseSPH[8] * (n.x * n.x - n.y * n.y);
    return max(result, vec3(0.0));
}

vec3 computeAnisotropicBentNormal(const in vec3 normal, const in vec3 eyeVector, const in float roughness, const in vec3 anisotropicT, const in vec3 anisotropicB, const in float anisotropy) {
    vec3 anisotropyDirection = anisotropy >= 0.0 ? anisotropicB : anisotropicT;
    vec3 anisotropicTangent = cross(anisotropyDirection, eyeVector);
    vec3 anisotropicNormal = cross(anisotropicTangent, anisotropyDirection);
    float bendFactor = abs(anisotropy) * clamp(5.0 * roughness, 0.0, 1.0);
    return normalize(mix(normal, anisotropicNormal, bendFactor));
}

const mat3 LUVInverse = mat3( 6.0013, -2.700, -1.7995, -1.332, 3.1029, -5.7720, 0.3007, -1.088, 5.6268 );
vec3 LUVToRGB( const in vec4 vLogLuv ) {
    float Le = vLogLuv.z * 255.0 + vLogLuv.w;
    vec3 Xp_Y_XYZp;
    Xp_Y_XYZp.y = exp2((Le - 127.0) / 2.0);
    Xp_Y_XYZp.z = Xp_Y_XYZp.y / vLogLuv.y;
    Xp_Y_XYZp.x = vLogLuv.x * Xp_Y_XYZp.z;
    vec3 vRGB = LUVInverse * Xp_Y_XYZp;
    return max(vRGB, 0.0);
}

float linRoughnessToMipmap(const in float roughnessLinear) {
    return sqrt(roughnessLinear);
}

vec3 prefilterEnvMapCube(const in float rLinear, const in vec3 R) {
    vec3 dir = R;
    vec2 uTextureEnvironmentSpecularPBRLodRange = vec2(2,5);
    vec2 uTextureEnvironmentSpecularPBRTextureSize = vec2(256,256);
    float lod = min(uTextureEnvironmentSpecularPBRLodRange.x, linRoughnessToMipmap(rLinear) * uTextureEnvironmentSpecularPBRLodRange.y);
    float scale = 1.0 - exp2(lod) / uTextureEnvironmentSpecularPBRTextureSize.x;
    vec3 absDir = abs(dir);
    float M = max(max(absDir.x, absDir.y), absDir.z);
    if (absDir.x != M) dir.x *= scale;
    if (absDir.y != M) dir.y *= scale;
    if (absDir.z != M) dir.z *= scale;
    return LUVToRGB(texture(u_SpecularCube, dir, lod));
}
vec3 getSpecularDominantDir(const in vec3 N, const in vec3 R, const in float realRoughness) {
    float smoothness = 1.0 - realRoughness;
    float lerpFactor = smoothness * (sqrt(smoothness) + realRoughness);
    return mix(N, R, lerpFactor);
}
vec3 getPrefilteredEnvMapColor(const in vec3 normal, const in vec3 eyeVector, const in float roughness, const in vec3 frontNormal) {
    vec3 R = reflect(-eyeVector, normal);
    R = getSpecularDominantDir(normal, R, roughness);
    vec3 prefilteredColor = prefilterEnvMapCube(roughness, uEnvironmentTransform * R);
    float factor = clamp(1.0 + dot(R, frontNormal), 0.0, 1.0);
    prefilteredColor *= factor * factor;
    return prefilteredColor;
}

vec3 integrateBRDF(const in vec3 specular, const in float roughness, const in float NoV, const in float f90) {
    vec4 rgba = texture(u_IBLLUT, vec2(NoV, roughness));
    float b = (rgba[3] * 65280.0 + rgba[2] * 255.0);
    float a = (rgba[1] * 65280.0 + rgba[0] * 255.0);
    const float div = 1.0 / 65535.0;
    return (specular * a + b * f90) * div;
}

vec3 computeIBLSpecularUE4(const in vec3 normal, const in vec3 eyeVector, const in float roughness, const in vec3 specular, const in vec3 frontNormal, const in float f90) {
    float NoV = dot(normal, eyeVector);
    return getPrefilteredEnvMapColor(normal, eyeVector, roughness, frontNormal) * integrateBRDF(specular, roughness, NoV, f90);
}

vec3 computeLightSSS(
const in float dotNL, const in float attenuation, const in float thicknessFactor, const in vec3 translucencyColor, const in float translucencyFactor, const in float shadowDistance, const in vec3 diffuse, const in vec3 lightColor) {
    float wrap = clamp(0.3 - dotNL, 0., 1.);
    float thickness = max(0.0, shadowDistance / max(0.001, thicknessFactor));
    float finalAttenuation = translucencyFactor * attenuation * wrap;
    return finalAttenuation * lightColor * diffuse * exp(-thickness / max(translucencyColor, vec3(0.001)));
}

vec3 sRGBToLinear(const in vec3 color) {
    return vec3( color.r < 0.04045 ? color.r * (1.0 / 12.92) : pow((color.r + 0.055) * (1.0 / 1.055), 2.4), color.g < 0.04045 ? color.g * (1.0 / 12.92) : pow((color.g + 0.055) * (1.0 / 1.055), 2.4), color.b < 0.04045 ? color.b * (1.0 / 12.92) : pow((color.b + 0.055) * (1.0 / 1.055), 2.4));
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

void main()
{
    FragColor = vec4(0,0,0,1);

    vec4 DiffuseAO = texture(u_DiffuseAO, v_TexCoord);
    vec4 SpecularOpacity = texture(u_Main, v_TexCoord);

    SpecularOpacity.rgb = sRGBToLinear(SpecularOpacity.rgb);
    DiffuseAO.rgb = sRGBToLinear(DiffuseAO.rgb);
    ditheringMaskingDiscard(gl_FragCoord, 1, SpecularOpacity.a, 1, ZNearFar.zw);

    float materialRoughness = 1 -0.5118;
    vec3 eyeVector = normalize(ViewPos.xyz - v_WSCurrPos.xyz);
    vec3 normal = GetWorldNormal();
    vec4 prepGGX = precomputeGGX(normal, eyeVector, max(0.045, materialRoughness));

    vec3 materialSpecular = SpecularOpacity.rgb;
    float attenuation = 1;
    float materialF90 = clamp(50.0 * materialSpecular.g, 0.0, 1.0);

    vec3 lightDiffuse;
    vec3 lightSpecular;
    bool lighted;

    float anisotropy = 1;
    float uAnisotropyDirection = 0;
    vec3 tangent = normalize(vec3(v2f.tspace0.x, v2f.tspace1.x, v2f.tspace2.x));
    vec3 bitangent = normalize(vec3(v2f.tspace0.y, v2f.tspace1.y, v2f.tspace2.y));
    vec3 anisotropicT = normalize(mix(tangent.xyz, bitangent, uAnisotropyDirection));
    vec3 anisotropicB = normalize(mix(bitangent, -tangent.xyz, uAnisotropyDirection));
    vec3 bentAnisotropicNormal = computeAnisotropicBentNormal(normal, eyeVector, materialRoughness, anisotropicT, anisotropicB, anisotropy);

    vec3 diffuse;
    vec3 specular;

    diffuse = DiffuseAO.rgb * computeDiffuseSPH(normal);
    diffuse *= DiffuseAO.a;

    vec3 vViewNormal = mat3(transpose(inverse(View))) * normal;
    vec3 frontNormal = normalize(gl_FrontFacing ? vViewNormal : -vViewNormal);

    specular = computeIBLSpecularUE4(bentAnisotropicNormal, eyeVector, materialRoughness, materialSpecular, frontNormal, materialF90);
    float aoSpec = specularOcclusion(1, DiffuseAO.a, normal, eyeVector);
    specular *= aoSpec;
    // specular = ssr(specular, materialSpecular * aoSpec, materialRoughness, bentAnisotropicNormal, eyeVector);

    vec2 uvoffset[4];
    uvoffset[0] = vec2(0,0);
    uvoffset[1] = vec2(0.5,0);
    uvoffset[2] = vec2(0,0.5);
    uvoffset[3] = vec2(0.5,0.5);
    Light[DIRECTIONAL_LIGHTS_MAX+POINT_LIGHTS_MAX] Lights = PrepareLights();
    for(int i=0;i<DirectionalLightNum;i++)
    {
        float dotNL = dot(normal, -Lights[i].direction);
        if(dotNL<0) dotNL=0;

        
        // Shadow
        vec3 LSpos = v_vLSPos[i];

        float actualDistance = LSpos.z;
        float bias = max(0.05 * (1.0 - dotNL), 0.005);
        float shadow = 0.0;
        float shadowDistance = 0.0;
        vec2 texelSize = 1.0 / vec2(1024);
        for(int x = -1; x <= 1; ++x)
        {
            for(int y = -1; y <= 1; ++y)
            {
                float lightDistance = GetDepth(LSpos.xy, uvoffset[i], vec2(x, y) * texelSize);
                lightDistance = lightDistance * 2 - 1;
                shadow += actualDistance - bias > lightDistance ? 0.0 : 1.0;      
                if(actualDistance - bias > lightDistance) shadowDistance += actualDistance - bias - lightDistance;
            }    
        }
        shadow /= 9.0;
        shadowDistance /= 9/0;
        float shadowIntensity = shadow;

        computeLightLambertGGXAnisotropy(normal, eyeVector, dotNL, prepGGX, DiffuseAO.rgb, 
            materialSpecular, attenuation, Lights[i].color, -Lights[i].direction, 
            materialF90, anisotropicT, anisotropicB, anisotropy, lightDiffuse, lightSpecular, lighted);

        diffuse += lightDiffuse * shadowIntensity;
        specular += lightSpecular * shadowIntensity;
        diffuse += 0.1 * computeLightSSS(dotNL, attenuation, 0, vec3(1, 0.4120, 0.0465), 1, shadowDistance, DiffuseAO.rgb, Lights[i].color);

    }

    FragColor.xyz = diffuse + specular;
    // FragColor.xyz = texture(u_SpecularCube, normal).xyz;
    FragColor = EncodeRGBM(FragColor.xyz);
    
    vec3 vNormal = mat3(transpose(inverse(View))) * normal;
    Normal = vec4(normalize(vNormal)*0.5 + vec3(0.5,0.5,0.5), 1);;
    
    vec2 offset = v_currPos.xy/v_currPos.w - v_prevPos.xy/v_prevPos.w;
    UVOffset = vec4(offset * 0.5,1.0,1.0);
}