#include "SIByLpch.h"
#include "Renderer2D.h"

#include "Shader.h"
#include "Sibyl/Graphic/Geometry/TriangleMesh.h"

namespace SIByL
{
	struct Renderer2DStorage
	{
		Ref<TriangleMesh> QuadMesh = nullptr;
		Ref<Shader> FlatColorShader = nullptr;
		Ref<Shader> TextureShader = nullptr;
		Ref<Texture> TexCheckboard = nullptr;
	};

	static Renderer2DStorage* s_Data;

	void Renderer2D::Init()
	{
		s_Data = new Renderer2DStorage;

		VertexBufferLayout layout =
		{
			{ShaderDataType::Float3, "POSITION"},
			{ShaderDataType::Float2, "TEXCOORD"},
		};

		std::vector<ConstantBufferLayout> CBlayouts =
		{
			// ConstantBuffer1
			{
				{ShaderDataType::Mat4, "Model"},
				{ShaderDataType::Mat4, "View"},
				{ShaderDataType::Mat4, "Projection"},
				{ShaderDataType::Float4, "Color"},
			},
		};

		std::vector<ShaderResourceLayout> SRlayouts =
		{
			// ShaderResourceTable 1
			{
				{ShaderResourceType::Texture2D, "Main"},
			},
		};

		//s_Data->TexCheckboard = Texture2D::Create("checkboard.png");

		s_Data->TextureShader = Shader::Create("SIByL/Texture",
			ShaderDesc({ true,layout }),
			ShaderBinderDesc(CBlayouts, SRlayouts));

		struct VertexData
		{
			float position[3];
			float uv[2];
		};

		VertexData vertices[] = {
			0.5f, 0.5f, 0.0f,     1.0f,1.0f,
			0.5f, -0.5f, 0.0f,    1.0f,0.0f,
			-0.5f, -0.5f, 0.0f,   0.0f ,0.0f,
			-0.5f, 0.5f, 0.0f,	  0.0f,1.0f,
		};

		uint32_t indices[] = { // 注意索引从0开始! 
			0, 1, 3, // 第一个三角形
			1, 2, 3  // 第二个三角形
		};

		s_Data->QuadMesh = TriangleMesh::Create((float*)vertices, 4, indices, 6, layout);
	}
	
	void Renderer2D::Shutdown()
	{
		delete s_Data;
	}

	void Renderer2D::BeginScene(Ref<Camera> camera)
	{
		s_Data->TextureShader->Use();
		s_Data->TextureShader->GetBinder()->SetMatrix4x4("View", camera->GetViewMatrix());
		s_Data->TextureShader->GetBinder()->SetMatrix4x4("Projection", camera->GetProjectionMatrix());
	}

	void Renderer2D::EndScene()
	{

	}

	void Renderer2D::DrawQuad(const glm::vec2& position, const glm::vec2& size, const glm::vec4& color)
	{

	}

	void Renderer2D::DrawQuad(const glm::vec3& position, const glm::vec2& size, const glm::vec4& color)
	{

	}

	void Renderer2D::DrawQuad(const glm::vec2& position, const glm::vec2& size, Ref<Texture2D> texture)
	{
		glm::mat4 model = glm::mat4(1.0f);

		model = glm::translate(model, { position,0 });
	}

	void Renderer2D::DrawQuad(const glm::vec3& position, const glm::vec2& size, Ref<Texture2D> texture)
	{
		glm::mat4 model = glm::mat4(1.0f);

		model = glm::scale(model, { size, 1 });
		model = glm::translate(model, { position });

		s_Data->TextureShader->Use();
		s_Data->TextureShader->GetBinder()->SetMatrix4x4("Model", model);
		s_Data->TextureShader->GetBinder()->TEMPUpdateAllConstants();

		s_Data->TextureShader->GetBinder()->SetTexture2D("Main", texture);
		s_Data->TextureShader->GetBinder()->TEMPUpdateAllResources();

		s_Data->QuadMesh->RasterDraw();
	}
}