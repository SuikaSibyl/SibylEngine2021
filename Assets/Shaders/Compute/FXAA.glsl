//////////////////////////////////////////////////////////////////////
/////                       Compute Shader                      //////
//////////////////////////////////////////////////////////////////////
#version 450 core
layout (local_size_x = 16, local_size_y = 16) in;
layout(rgba8, binding = 0) uniform image2D img_output;

layout(std430, binding=0) buffer Input
{
    vec2 OutputSize;
};

layout(binding = 0) uniform sampler2D u_Texture;

vec4 fxaa(vec2 uv) {
    vec2 fxaaQualityRcpFrame = vec2(1.0) / gl_NumWorkGroups.xy;
    float fxaaQualitySubpix = 0.75;
    float fxaaQualityEdgeThreshold = 0.125;
    float fxaaQualityEdgeThresholdMin = 0.0625;

    vec2 posM = uv;
    vec2 maxCoord = vec2(1.0) - vec2(1.0) / gl_NumWorkGroups.xy;
    vec2 minCoord = vec2(1.0) / gl_NumWorkGroups.xy;

    vec4 rgbyM = (texture(u_Texture, posM));
    float lumaM = rgbyM.y;
    vec4 sw = vec4(-1.0, 1.0, 1.0, -1.0) * fxaaQualityRcpFrame.xxyy;
    
    float lumaS = (texture(u_Texture, clamp(posM + vec2(0.0, sw.z), minCoord, maxCoord))).y;
    float lumaE = (texture(u_Texture, clamp(posM + vec2(sw.y, 0.0), minCoord, maxCoord))).y;
    float lumaN = (texture(u_Texture, clamp(posM + vec2(0.0, sw.w), minCoord, maxCoord))).y;
    float lumaW = (texture(u_Texture, clamp(posM + vec2(sw.x, 0.0), minCoord, maxCoord))).y;

    float maxSM = max(lumaS, lumaM);
    float minSM = min(lumaS, lumaM);
    float maxESM = max(lumaE, maxSM);
    float minESM = min(lumaE, minSM);
    float maxWN = max(lumaN, lumaW);
    float minWN = min(lumaN, lumaW);
    float rangeMax = max(maxWN, maxESM);
    float rangeMin = min(minWN, minESM);
    float rangeMaxScaled = rangeMax * fxaaQualityEdgeThreshold;
    float range = rangeMax - rangeMin;
    float rangeMaxClamped = max(fxaaQualityEdgeThresholdMin, rangeMaxScaled);
    bool earlyExit = range < rangeMaxClamped;
    if (earlyExit) {
        return rgbyM;
    }
    
    float lumaNW = (texture(u_Texture, clamp(posM + sw.xw, minCoord, maxCoord))).y;
    float lumaSE = (texture(u_Texture, clamp(posM + sw.yz, minCoord, maxCoord))).y;
    float lumaNE = (texture(u_Texture, clamp(posM + sw.yw, minCoord, maxCoord))).y;
    float lumaSW = (texture(u_Texture, clamp(posM + sw.xy, minCoord, maxCoord))).y;

    float lumaNS = lumaN + lumaS;
    float lumaWE = lumaW + lumaE;
    float subpixRcpRange = 1.0/range;
    float subpixNSWE = lumaNS + lumaWE;
    float edgeHorz1 = (-2.0 * lumaM) + lumaNS;
    float edgeVert1 = (-2.0 * lumaM) + lumaWE;
    float lumaNESE = lumaNE + lumaSE;
    float lumaNWNE = lumaNW + lumaNE;
    float edgeHorz2 = (-2.0 * lumaE) + lumaNESE;
    float edgeVert2 = (-2.0 * lumaN) + lumaNWNE;
    float lumaNWSW = lumaNW + lumaSW;
    float lumaSWSE = lumaSW + lumaSE;
    float edgeHorz4 = (abs(edgeHorz1) * 2.0) + abs(edgeHorz2);
    float edgeVert4 = (abs(edgeVert1) * 2.0) + abs(edgeVert2);
    float edgeHorz3 = (-2.0 * lumaW) + lumaNWSW;
    float edgeVert3 = (-2.0 * lumaS) + lumaSWSE;
    float edgeHorz = abs(edgeHorz3) + edgeHorz4;
    float edgeVert = abs(edgeVert3) + edgeVert4;
    float subpixNWSWNESE = lumaNWSW + lumaNESE;
    float lengthSign = fxaaQualityRcpFrame.x;
    bool horzSpan = edgeHorz >= edgeVert;
    float subpixA = subpixNSWE * 2.0 + subpixNWSWNESE;
    if(!horzSpan) lumaN = lumaW;
    if(!horzSpan) lumaS = lumaE;
    if(horzSpan) lengthSign = fxaaQualityRcpFrame.y;
    float subpixB = (subpixA * (1.0/12.0)) - lumaM;
    float gradientN = lumaN - lumaM;
    float gradientS = lumaS - lumaM;
    float lumaNN = lumaN + lumaM;
    float lumaSS = lumaS + lumaM;
    bool pairN = abs(gradientN) >= abs(gradientS);
    float gradient = max(abs(gradientN), abs(gradientS));
    if(pairN) lengthSign = -lengthSign;
    float subpixC = clamp(abs(subpixB) * subpixRcpRange, 0.0, 1.0);
    vec2 posB = posM.xy;
    vec2 offNP;
    offNP.x = (!horzSpan) ? 0.0 : fxaaQualityRcpFrame.x;
    offNP.y = ( horzSpan) ? 0.0 : fxaaQualityRcpFrame.y;
    if(!horzSpan) posB.x += lengthSign * 0.5;
    if( horzSpan) posB.y += lengthSign * 0.5;
    vec2 posN;
    posN.x = posB.x - offNP.x;
    posN.y = posB.y - offNP.y;
    vec2 posP;
    posP.x = posB.x + offNP.x;
    posP.y = posB.y + offNP.y;
    float subpixD = ((-2.0)*subpixC) + 3.0;
    float lumaEndN = (texture(u_Texture, clamp(posN, minCoord, maxCoord))).y;
    float subpixE = subpixC * subpixC;
    float lumaEndP = (texture(u_Texture, clamp(posP, minCoord, maxCoord))).y;
    if(!pairN) lumaNN = lumaSS;
    float gradientScaled = gradient * 1.0/4.0;
    float lumaMM = lumaM - lumaNN * 0.5;
    float subpixF = subpixD * subpixE;
    bool lumaMLTZero = lumaMM < 0.0;
    lumaEndN -= lumaNN * 0.5;
    lumaEndP -= lumaNN * 0.5;
    bool doneN = abs(lumaEndN) >= gradientScaled;
    bool doneP = abs(lumaEndP) >= gradientScaled;
    if(!doneN) posN.x -= offNP.x * 1.5;
    if(!doneN) posN.y -= offNP.y * 1.5;
    bool doneNP = (!doneN) || (!doneP);
    if(!doneP) posP.x += offNP.x * 1.5;
    if(!doneP) posP.y += offNP.y * 1.5;
    if(doneNP) {        
        if(!doneN) lumaEndN = (texture(u_Texture, clamp(posN.xy, minCoord, maxCoord))).y;
        if(!doneP) lumaEndP = (texture(u_Texture, clamp(posP.xy, minCoord, maxCoord))).y;

        if(!doneN) lumaEndN = lumaEndN - lumaNN * 0.5;
        if(!doneP) lumaEndP = lumaEndP - lumaNN * 0.5;
        doneN = abs(lumaEndN) >= gradientScaled;
        doneP = abs(lumaEndP) >= gradientScaled;
        if(!doneN) posN.x -= offNP.x * 2.0;
        if(!doneN) posN.y -= offNP.y * 2.0;
        doneNP = (!doneN) || (!doneP);
        if(!doneP) posP.x += offNP.x * 2.0;
        if(!doneP) posP.y += offNP.y * 2.0;
        if(doneNP) {
            if(!doneN) lumaEndN = (texture(u_Texture, clamp(posN.xy, minCoord, maxCoord))).y;
            if(!doneP) lumaEndP = (texture(u_Texture, clamp(posP.xy, minCoord, maxCoord))).y;
            if(!doneN) lumaEndN = lumaEndN - lumaNN * 0.5;
            if(!doneP) lumaEndP = lumaEndP - lumaNN * 0.5;
            doneN = abs(lumaEndN) >= gradientScaled;
            doneP = abs(lumaEndP) >= gradientScaled;
            if(!doneN) posN.x -= offNP.x * 4.0;
            if(!doneN) posN.y -= offNP.y * 4.0;
            doneNP = (!doneN) || (!doneP);
            if(!doneP) posP.x += offNP.x * 4.0;
            if(!doneP) posP.y += offNP.y * 4.0;
            if(doneNP) {
                if(!doneN) lumaEndN = (texture(u_Texture, clamp(posN.xy, minCoord, maxCoord))).y;
                if(!doneP) lumaEndP = (texture(u_Texture, clamp(posP.xy, minCoord, maxCoord))).y;
                if(!doneN) lumaEndN = lumaEndN - lumaNN * 0.5;
                if(!doneP) lumaEndP = lumaEndP - lumaNN * 0.5;
                doneN = abs(lumaEndN) >= gradientScaled;
                doneP = abs(lumaEndP) >= gradientScaled;
                if(!doneN) posN.x -= offNP.x * 12.0;
                if(!doneN) posN.y -= offNP.y * 12.0;
                doneNP = (!doneN) || (!doneP);
                if(!doneP) posP.x += offNP.x * 12.0;
                if(!doneP) posP.y += offNP.y * 12.0;
            }

        }

    }
    float dstN = posM.x - posN.x;
    float dstP = posP.x - posM.x;
    if(!horzSpan) dstN = posM.y - posN.y;
    if(!horzSpan) dstP = posP.y - posM.y;
    bool goodSpanN = (lumaEndN < 0.0) != lumaMLTZero;
    float spanLength = (dstP + dstN);
    bool goodSpanP = (lumaEndP < 0.0) != lumaMLTZero;
    float spanLengthRcp = 1.0/spanLength;
    bool directionN = dstN < dstP;
    float dst = min(dstN, dstP);
    bool goodSpan = directionN ? goodSpanN : goodSpanP;
    float subpixG = subpixF * subpixF;
    float pixelOffset = (dst * (-spanLengthRcp)) + 0.5;
    float subpixH = subpixG * fxaaQualitySubpix;
    float pixelOffsetGood = goodSpan ? pixelOffset : 0.0;
    float pixelOffsetSubpix = max(pixelOffsetGood, subpixH);
    if(!horzSpan) posM.x += pixelOffsetSubpix * lengthSign;
    if( horzSpan) posM.y += pixelOffsetSubpix * lengthSign;
    return  (texture(u_Texture, clamp(posM, minCoord, maxCoord)));
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

    vec4 fxaaCol = fxaa(vec2(u,v));
    pixel= vec4(fxaaCol.rgb, 1);

    // output to a specific pixel in the image
    imageStore(img_output, pixel_coords, pixel);
}
