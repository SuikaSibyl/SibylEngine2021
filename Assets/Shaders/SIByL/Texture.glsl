//////////////////////////////////////////////////////////////////////
/////                       Vertex Shader                       //////
//////////////////////////////////////////////////////////////////////
#type VS
#version 330 core

// VData Inputs
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aUV;

// Vertex Outputs
out vec3 v_Color;
out vec2 v_TexCoord;

// Uniform Constants
uniform mat4 Model;
uniform mat4 Projection;
uniform mat4 View;
uniform vec4 Color;

void main()
{
    gl_Position = Projection* View * Model * vec4(aPos, 1.0);
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
// Fragment outputs
out vec4 FragColor;  
// Uniform items
uniform vec4 Color;
uniform sampler2D u_Texture;

void main()
{
    FragColor = Color * texture(u_Texture, v_TexCoord);
}