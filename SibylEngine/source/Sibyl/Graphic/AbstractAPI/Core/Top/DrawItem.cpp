#include "SIByLpch.h"
#include "DrawItem.h"

#include <Sibyl/Graphic/AbstractAPI/Core/Middle/Shader.h>
#include <Sibyl/Graphic/AbstractAPI/Core/Middle/ShaderBinder.h>
#include <Sibyl/Graphic/AbstractAPI/Core/Top/Material.h>
#include "Sibyl/Graphic/Core/Geometry/TriangleMesh.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Graphic.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Material.h"

namespace SIByL
{
	DrawItem::DrawItem()
		:m_Mesh(nullptr), m_SubMesh(nullptr)
	{

	}


	DrawItem::DrawItem(Ref<TriangleMesh> mesh)
		:m_Mesh(mesh), m_SubMesh(&(*mesh->begin()))
	{

	}

	DrawItem::DrawItem(Ref<TriangleMesh> mesh, SubMesh* submesh)
		:m_Mesh(mesh), m_SubMesh(submesh)
	{

	}

	DrawItem::~DrawItem()
	{

	}

	void DrawItem::SetObjectMatrix(const glm::mat4& transform)
	{
		m_ConstantsBuffer->SetMatrix4x4("Model", transform);
	}

	void DrawItem::OnDrawCall()
	{
		Graphic::CurrentMaterial->m_Shader->GetBinder()->BindConstantsBuffer(0, *m_ConstantsBuffer);
		m_ConstantsBuffer->UploadDataIfDirty(Graphic::CurrentMaterial->m_Shader->GetBinder().get());
		m_Mesh->RasterDrawSubmeshStart();
		m_Mesh->RasterDrawSubmesh(*m_SubMesh);
	}

}