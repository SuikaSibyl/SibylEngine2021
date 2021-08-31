SamplerState gSamPointWrap : register(s0);
SamplerState gSamPointClamp : register(s1);
SamplerState gSamLinearWarp : register(s2);
SamplerState gSamLinearClamp : register(s3);
SamplerState gSamAnisotropicWarp : register(s4);
SamplerState gSamAnisotropicClamp : register(s5);

Texture2D gDiffuseMap: register(t0);//所有漫反射贴图


cbuffer cbPerObject : register(b0)
{
	float4x4 Model;
	float4x4 View;
	float4x4 Projection;
	float3 color;
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
	
	float4x4 VP = mul(Projection, View);
	// float4x4 MVP = mul(Model, VP);
	// vout.PosH = mul(View, float4(vin.PosL, 1.0f));
	vout.PosH = mul(View, float4(vin.PosL, 1.0f));
	vout.PosH = mul(Projection, vout.PosH);

	// Just pass vertex color into the pixel shader.
    vout.Color = float4(color, 1.0);
    vout.UV = vin.UV;
    return vout;
}

float4 PS(VertexOut pin) : SV_Target
{
	float4 color = gDiffuseMap.Sample(gSamLinearWarp, pin.UV);
    return color*pin.Color;
}