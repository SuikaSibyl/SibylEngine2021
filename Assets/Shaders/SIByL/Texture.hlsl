SamplerState gSamPointWrap : register(s0);
SamplerState gSamPointClamp : register(s1);
SamplerState gSamLinearWarp : register(s2);
SamplerState gSamLinearClamp : register(s3);
SamplerState gSamAnisotropicWarp : register(s4);
SamplerState gSamAnisotropicClamp : register(s5);

Texture2D gDiffuseMap: register(t0);

cbuffer cbPerObject : register(b0)
{
	float4x4 Model;
};

cbuffer cbPerMaterial : register(b1)
{
	float4   Color;
};

cbuffer cbPerCamera : register(b2)
{
	float4x4 View;
	float4x4 Projection;
};

cbuffer cbPerFrame : register(b3)
{
	float4 LightColor;
};

struct VertexIn
{
	float3 PosL  : POSITION;
	float2 UV	: TEXCOORD;
};

struct VertexOut
{
	float4 PosH  : SV_POSITION;
    float4 Color : COLOR;
	float2 UV	: POSITION;
};

VertexOut VS(VertexIn vin)
{
	VertexOut vout;
	
	vout.PosH = mul(Model, float4(vin.PosL, 1.0f));
	vout.PosH = mul(View, vout.PosH);
	vout.PosH = mul(Projection, vout.PosH);

    vout.Color = Color;
    vout.UV = vin.UV;
    return vout;
}

struct PS_OUTPUT
{
    float4 Color: SV_Target0;
    float4 Normal: SV_Target1;
};

PS_OUTPUT PS(VertexOut pin) : SV_Target
{
	PS_OUTPUT output;
	float4 color = gDiffuseMap.Sample(gSamLinearWarp, pin.UV);
	output.Color = pin.Color * color;
	output.Normal = float4(1,0,0,1);
    return output;
}