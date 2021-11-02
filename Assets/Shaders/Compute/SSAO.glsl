//////////////////////////////////////////////////////////////////////
/////                       Compute Shader                      //////
//////////////////////////////////////////////////////////////////////
#version 450 core
layout (local_size_x = 16, local_size_y = 16) in;
layout(rgba8, binding = 0) uniform image2D img_output;

layout(std430, binding=0) buffer Input
{
    mat4  uProjection;
    vec2 OutputSize;
    float uRadius;
};

layout(binding = 0) uniform sampler2D u_Normal;
layout(binding = 0) uniform sampler2D u_Depth;

// Fetch Frag Pos in Vies Space
vec3 GetFragPos(float depth, vec2 uv)
{
    float zc = depth * (100 - 0.001) + 0.001;
    uv = -2 * (uv-vec2(0.5,0.5));
    float TANFOVD2 = 0.414213562373;
    return vec3(-uv.x*TANFOVD2*zc *gl_NumWorkGroups.x/ gl_NumWorkGroups.y, -uv.y*TANFOVD2*zc ,zc);
}   

vec2 Pos2UV(vec3 pos)
{
    vec2 xy = pos.xy / pos.z;
    float TANFOVD2 = 0.414213562373;
    float v = xy.y / TANFOVD2;
    float u = xy.x / (TANFOVD2 * gl_NumWorkGroups.x/ gl_NumWorkGroups.y);
    vec2 uv = vec2(u,v);
    uv += vec2(1,1);
    uv/=2;
    return uv;
}   

float decode24(const in vec3 x) {
    const vec3 decode = 1.0 / vec3(1.0, 255.0, 65025.0);
    return dot(x, decode);
}

// Fetch Frag Depth (in NDC space, range in [0,1])
// uv range in [0,1]
float GetFragDepth(vec2 uv)
{
    // if(uv.y<0 || uv.y > 1)
    // {
    //     return 100;
    // }
    float depth = decode24(texture(u_Depth, uv).rgb);
    float zc = depth * (100 - 0.001) + 0.001;
    return zc;
}

vec3 SSAO(const in vec2 uv)
{
    vec3 samples[] =
    {
        vec3(0.497709, -0.447092, 0.499634),
        vec3(0.145427, 0.164949, 0.0223372),
        vec3(-0.402936, -0.19206, 0.316554),
        vec3(0.135109, -0.89806, 0.401306),
        vec3(0.540875, 0.577609, 0.557006),
        vec3(0.874615, 0.41973, 0.146465),
        vec3(-0.0188978, -0.504141, 0.618431),
        vec3(-0.00298402, -0.00169127, 0.00333421),
        vec3(0.438746, -0.408985, 0.222553),
        vec3(0.323672, 0.266571, 0.27902),
        vec3(-0.261392, 0.167732, 0.184589),
        vec3(0.440034, -0.292085, 0.430474),
        vec3(0.435821, -0.171226, 0.573847),
        vec3(-0.117331, -0.0274799, 0.40452),
        vec3(-0.174974, -0.173549, 0.174403),
        vec3(-0.22543, 0.143145, 0.169986),
        vec3(-0.112191, 0.0920681, 0.0342291),
        vec3(0.448674, 0.685331, 0.0673666),
        vec3(-0.257349, -0.527384, 0.488827),
        vec3(-0.464402, -0.00938766, 0.473935),
        vec3(-0.0553817, -0.174926, 0.102575),
        vec3(0.0163094, -0.0247947, 0.0211469),
        vec3(-0.0357804, -0.319047, 0.326624),
        vec3(0.435365, -0.0369896, 0.662937),
        vec3(0.339125, 0.56041, 0.472273),
        vec3(0.00165474, 0.00189482, 0.00127085),
        vec3(-0.421643, 0.263322, 0.409346),
        vec3(-0.0171094, -0.459828, 0.622265),
        vec3(-0.273823, 0.126528, 0.823235),
        vec3(-0.00968538, 0.0108071, 0.0102621),
        vec3(-0.364436, 0.478037, 0.558969),
        vec3(0.15067, 0.333067, 0.191465),
        vec3(0.414059, -0.0692679, 0.401582),
        vec3(-0.484817, -0.458746, 0.367069),
        vec3(-0.530125, -0.589921, 0.16319),
        vec3(-0.118435, 0.235465, 0.202611),
        vec3(-0.00666287, -0.0052001, 0.010577),
        vec3(-0.241253, -0.454733, 0.747212),
        vec3(-0.541038, 0.757421, 0.213657),
        vec3(-0.0633459, 0.66141, 0.73048),
        vec3(0.458887, -0.599781, 0.24389),
        vec3(0.116971, 0.222313, 0.688396),
        vec3(-0.268377, 0.244657, 0.574693),
        vec3(0.304252, -0.129121, 0.453988),
        vec3(0.100759, -0.433708, 0.282605),
        vec3(-0.343713, -0.0738141, 0.0292256),
        vec3(0.251075, 0.0834831, 0.238692),
        vec3(-0.0756226, 0.0950082, 0.0954248),
        vec3(-0.0389006, -0.133558, 0.361451),
        vec3(-0.226506, 0.315615, 0.00827583),
        vec3(0.244327, 0.354923, 0.0673253),
        vec3(0.0447351, 0.568618, 0.243966),
        vec3(0.119581, -0.446107, 0.0971173),
        vec3(0.316438, -0.328146, 0.270037),
        vec3(0.51475, 0.448266, 0.714832),
        vec3(-0.727464, 0.385414, 0.393764),
        vec3(0.537968, 0.00715645, 0.149009),
        vec3(0.450305, 0.00440889, 0.105299),
        vec3(0.39208, 0.0368202, 0.212718),
        vec3(-0.0958963, 0.592978, 0.0653918),
        vec3(0.973455, -0.00306814, 0.112386),
        vec3(0.496669, -0.841329, 0.00418623),
        vec3(0.441751, -0.163923, 0.489625),
        vec3(-0.455431, -0.698782, 0.191856),
    };

    vec3 noises[] = 
    {
        vec3(-0.729046, 0.629447, 0),
        vec3(0.670017, 0.811584, 0),
        vec3(0.937736, -0.746026, 0),
        vec3(-0.557932, 0.826752, 0),
        vec3(-0.383666, 0.264719, 0),
        vec3(0.0944412, -0.804919, 0),
        vec3(-0.623236, -0.443004, 0),
        vec3(0.985763, 0.093763, 0),
        vec3(0.992923, 0.915014, 0),
        vec3(0.93539, 0.929777, 0),
        vec3(0.451678, -0.684774, 0),
        vec3(0.962219, 0.941186, 0),
        vec3(-0.780276, 0.914334, 0),
        vec3(0.596212, -0.0292487, 0),
        vec3(-0.405941, 0.600561, 0),
        vec3(-0.990433, -0.716227, 0),
    };

    vec3 normal = texture(u_Normal, uv).rgb;
    normal = (normal-0.5)*2;
    normal = normalize(normal);
    vec4 pack = texture(u_Depth, uv);
    if(pack.a == 0)
    {
        return vec3(1,1,1);
    }
    float depth = decode24(pack.rgb);
    vec3 fragPos = GetFragPos(depth, uv);

    uint unoise = gl_GlobalInvocationID.x;
    uint vnoise = gl_GlobalInvocationID.y;
    vec3 randomVec = noises[(unoise%4)*4 + (vnoise%4)].xzy;

    vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);

    float occlusion = 0.0;
    vec3 indirect = vec3(0,0,0);
    
    for(int i = 0; i < 32; ++i)
    {
        vec3 samp = TBN * samples[i];
        vec3 sample_pos = fragPos + samp * .5; 
        vec2 offsetuv = Pos2UV(sample_pos);
        float sampleDepth = GetFragDepth(offsetuv);
        float distance = (abs(sample_pos.z - sampleDepth)*length(sample_pos)/abs(sample_pos.z));
        float range = .5 / distance;
        float rangeCheck = smoothstep(0.0, 1.0, range);
        if(rangeCheck<0.9) rangeCheck=0.0;
        occlusion += (sampleDepth <= sample_pos.z ? 1.0 : 0.0)* rangeCheck;
    }

    return vec3(1 - occlusion/32);
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

    vec3 occlusion = SSAO(vec2(u,v));
    pixel= vec4(occlusion, 1);

    // output to a specific pixel in the image
    imageStore(img_output, pixel_coords, pixel);
}
