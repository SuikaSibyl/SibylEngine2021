#include "SIByLpch.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "Renderer2D.h"

#include "Sibyl/Graphic/AbstractAPI/Core/Middle/Shader.h"
#include "Sibyl/Graphic/Core/Geometry/TriangleMesh.h"
#include "Sibyl/Graphic/Core/Texture/Image.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Material.h"

#include "Sibyl/Graphic/AbstractAPI/Core/Top/Graphic.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/DrawItem.h"
#include "Sibyl/Graphic/Core/Geometry/MeshLoader.h"

namespace SIByL
{
	struct Renderer2DStorage
	{
		Ref<TriangleMesh> QuadMesh = nullptr;
		Ref<Shader> FlatColorShader = nullptr;
		Ref<Shader> TextureShader = nullptr;
		Ref<Texture2D> TexCheckboard = nullptr;
		Ref<Texture2D> TexWhite = nullptr;
		Ref<Image> WhiteImage = nullptr;
		Ref<Material> DefaultMaterial = nullptr;
		Ref<Camera> Camera = nullptr;
		Ref<DrawItem> DrawItem = nullptr;
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
			// ConstantBuffer0 : Per Object
			ConstantBufferLayout::PerObjectConstants,
			// ConstantBuffer1 : Per Material
			{
				{ShaderDataType::RGBA, "Color"},
			},
			// ConstantBuffer2 : Per Camera
			ConstantBufferLayout::PerCameraConstants,
			// ConstantBuffer3 : Per Frame
			ConstantBufferLayout::PerFrameConstants,
		};

		std::vector<ShaderResourceLayout> SRlayouts =
		{
			// ShaderResourceTable 1
			{
				{ShaderResourceType::Texture2D, "Main"},
			},
		};

		//s_Data->TexCheckboard = Texture2D::Create("checkboard.png");

		s_Data->TextureShader = Shader::Create("Shaders/SIByL/Texture",
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

		s_Data->WhiteImage = CreateRef<Image>(16, 16, 4, glm::vec4{ 1,1,1,1 });
		s_Data->TexWhite = Texture2D::Create(s_Data->WhiteImage, "White");
		s_Data->DefaultMaterial = CreateRef<Material>(s_Data->TextureShader);

		s_Data->DefaultMaterial->SetFloat4("Color", { 1,0,0,1 });
		s_Data->DefaultMaterial->SetTexture2D("Main", s_Data->TexWhite);
		s_Data->DrawItem = CreateRef<DrawItem>(s_Data->QuadMesh);
	}

	Ref<Material> Renderer2D::GetMaterial()
	{
		return s_Data->DefaultMaterial;
	}

	Ref<TriangleMesh> Renderer2D::GetMesh()
	{
		return s_Data->QuadMesh;
	}

	void Renderer2D::Shutdown()
	{
		delete s_Data;
	}
	
	void Renderer2D::BeginScene(Ref<Camera> camera)
	{
		//s_Data->TextureShader->Use();
		//s_Data->TextureShader->GetBinder()->SetMatrix4x4("View", camera->GetViewMatrix());
		//s_Data->TextureShader->GetBinder()->SetMatrix4x4("Projection", camera->GetProjectionMatrix());
		s_Data->Camera = camera;
	}

	void Renderer2D::EndScene()
	{

	}

	void Renderer2D::DrawQuad(const glm::mat4& transform, Ref<Material> material)
	{
		s_Data->Camera->SetCamera();

		material->SetPass();

		s_Data->DrawItem->SetObjectMatrix(transform);

		Graphic::CurrentCamera->OnDrawCall();
		Graphic::CurrentMaterial->OnDrawCall();
		s_Data->DrawItem->OnDrawCall();
	}

	void Renderer2D::DrawQuad(const glm::mat4& transform, const glm::vec4& color, Ref<Texture2D> texture)
	{
		//s_Data->TextureShader->Use();
		//s_Data->TextureShader->GetBinder()->SetMatrix4x4("Model", transform);
		//s_Data->TextureShader->GetBinder()->SetFloat4("Color", color);
		//s_Data->TextureShader->GetBinder()->SetTexture2D("Main", texture);
		//s_Data->QuadMesh->RasterDraw();
	}
}