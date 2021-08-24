cbuffer cbPerObject : register(b0)
{
	float3 color;
};

struct VertexIn
{
	float3 PosL  : POSITION;
};

struct VertexOut
{
	float4 PosH  : SV_POSITION;
    float4 Color : COLOR;
};

VertexOut VS(VertexIn vin)
{
	VertexOut vout;
	
	vout.PosH = float4(vin.PosL, 1.0f);
	
	// Just pass vertex color into the pixel shader.
    vout.Color = float4(color, 1.0);
    
    return vout;
}

float4 PS(VertexOut pin) : SV_Target
{
    return float4(1,1,1,1);
}