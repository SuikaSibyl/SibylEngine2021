#include "SIByLpch.h"
#include "MeshFilter.h"
#include "MeshRenderer.h"

#include "Sibyl/Graphic/Core/Geometry/TriangleMesh.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/DrawItem.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/ShaderBinder.h"

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

	MeshFilterComponent::MeshFilterComponent()
	{
		PerObjectBuffer = PerObjectConstantsBufferPool::GetPerObjectConstantsBuffer();
	}

	MeshFilterComponent::~MeshFilterComponent()
	{
		PerObjectConstantsBufferPool::PushToPool(PerObjectBuffer);
	}

	UINT MeshFilterComponent::GetSubmeshNum()
	{
		return Mesh->GetSubmesh();
	}

}