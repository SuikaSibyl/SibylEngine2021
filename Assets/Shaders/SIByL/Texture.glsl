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
out vec3 v_Color;
out vec2 v_TexCoord;
out vec4 v_currPos;
out vec4 v_prevPos;

// Uniform Constants
uniform mat4 Model;
uniform mat4 View;
uniform mat4 Projection;
uniform mat4 PreviousPV;
uniform mat4 CurrentPV;
uniform vec4 Color;

void main()
{
    gl_Position = Projection * View * Model * vec4(aPos, 1.0);
    v_currPos = CurrentPV * Model * vec4(aPos, 1.0);
    v_prevPos = PreviousPV * Model * vec4(aPos, 1.0);

    v_Color = aPos;
    v_TexCoord = aUV;
}

//////////////////////////////////////////////////////////////////////
/////                     Fragment Shader                       //////
//////////////////////////////////////////////////////////////////////
#type FS
#version 330 core
// Vertex Inputs
in vec3 v_Color;
in vec2 v_TexCoord;
in vec4 v_currPos;
in vec4 v_prevPos;
// Fragment outputs
layout(location = 0) out vec4 FragColor;  
layout(location = 1) out vec4 UVOffset;  
// Uniform items
uniform vec4 Color;
uniform sampler2D u_Main;

void main()
{
    FragColor = Color * texture(u_Main, v_TexCoord);

    if(FragColor.a < 0.1)
        discard;

    vec2 offset = v_currPos.xy/v_currPos.w - v_prevPos.xy/v_prevPos.w;
    UVOffset = vec4(offset * 0.5,1.0,1.0);
}