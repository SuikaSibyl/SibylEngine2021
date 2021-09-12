#include "SIByLpch.h"
#include "DrawItem.h"

#include <Sibyl/Graphic/AbstractAPI/Core/Middle/Shader.h>
#include <Sibyl/Graphic/AbstractAPI/Core/Middle/ShaderBinder.h>
#include <Sibyl/Graphic/AbstractAPI/Core/Top/Material.h>
#include "Sibyl/Graphic/Core/Geometry/TriangleMesh.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Graphic.h"

namespace SIByL
{
	static ShaderConstantsDesc PerObjectConstantsDesc;
	static ShaderConstantsDesc* GetPerObjectConstantsDesc()
	{
		// Init if not Inited
		if (PerObjectConstantsDesc.Size == -1)
		{
			ConstantBufferLayout& layout = ConstantBufferLayout::PerObjectConstants;
			int paraIndex = 0;

			PerObjectConstantsDesc.Size = layout.GetStide();
			for (auto bufferElement : layout)
			{
				PerObjectConstantsDesc.Mapper.InsertConstant(bufferElement, 1);
			}
		}

		return &PerObjectConstantsDesc;
	}

	std::vector<Ref<ShaderConstantsBuffer>> PerObjectConstantsBufferPool::m_IdleConstantsBuffer;

	Ref<ShaderConstantsBuffer> PerObjectConstantsBufferPool::GetPerObjectConstantsBuffer()
	{
		Ref<ShaderConstantsBuffer> result;
		if (!m_IdleConstantsBuffer.empty())
		{
			result = m_IdleConstantsBuffer.back();
			m_IdleConstantsBuffer.pop_back();
		}
		else
		{
			result = ShaderConstantsBuffer::Create
				(GetPerObjectConstantsDesc());

		}
		return result;
	}
	
	void PerObjectConstantsBufferPool::PushToPool(Ref<ShaderConstantsBuffer> buffer)
	{
		m_IdleConstantsBuffer.push_back(buffer);
	}

	DrawItem::DrawItem(Ref<TriangleMesh> mesh)
		:m_Mesh(mesh)
	{
		m_ConstantsBuffer = PerObjectConstantsBufferPool::GetPerObjectConstantsBuffer();
	}

	DrawItem::~DrawItem()
	{
		PerObjectConstantsBufferPool::PushToPool(m_ConstantsBuffer);
	}

	void DrawItem::SetObjectMatrix(const glm::mat4& transform)
	{
		m_ConstantsBuffer->SetMatrix4x4("Model", transform);
	}

	void DrawItem::OnDrawCall()
	{
		Graphic::CurrentMaterial->m_Shader->GetBinder()->BindConstantsBuffer(0, *m_ConstantsBuffer);
		m_ConstantsBuffer->UploadDataIfDirty();
		m_ConstantsBuffer->UploadDataIfDirty();
		m_Mesh->RasterDrawSubmeshStart();
		for (SubMesh& iter : *m_Mesh)
		{
			m_Mesh->RasterDrawSubmesh(iter);
		}
	}

}