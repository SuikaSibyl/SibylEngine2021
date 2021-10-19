#include "SIByLpch.h"
#include "FrameConstantsManager.h"

#include "Graphic.h"

#include <Sibyl/Graphic/AbstractAPI/Core/Middle/Shader.h>
#include <Sibyl/Graphic/AbstractAPI/Core/Middle/ShaderBinder.h>
#include <Sibyl/Graphic/AbstractAPI/Core/Top/Material.h>

namespace SIByL
{
	static ShaderConstantsDesc PerFrameConstantsDesc;
	static ShaderConstantsDesc* GetPerFrameConstantsDesc()
	{
		// Init if not Inited
		if (PerFrameConstantsDesc.Size == -1)
		{
			ConstantBufferLayout& layout = ConstantBufferLayout::PerFrameConstants;
			int paraIndex = 0;

			PerFrameConstantsDesc.Size = layout.GetStide();
			for (auto bufferElement : layout)
			{
				PerFrameConstantsDesc.Mapper.InsertConstant(bufferElement, 2);
			}
		}

		return &PerFrameConstantsDesc;
	}

	FrameConstantsManager::FrameConstantsManager()
	{
		ShaderConstantsDesc* desc = GetPerFrameConstantsDesc();
		m_ConstantsBuffer = ShaderConstantsBuffer::Create(desc);
		m_ConstantsBuffer->SetFloat3("directionalLights[0].direction", glm::vec3(0, -1, 0));
		m_ConstantsBuffer->SetFloat3("directionalLights[0].color", glm::vec3(1, 1, 1));
		m_ConstantsBuffer->SetFloat("directionalLights[0].intensity", 1);
		m_ConstantsBuffer->SetInt("DirectionalLightNum", 1);
	}

	void FrameConstantsManager::SetFrame()
	{
		Graphic::CurrentFrameConstantsManager = this;
	}

	void FrameConstantsManager::OnDrawCall()
	{
		// Upload Per-Material parameters to GPU
		m_ConstantsBuffer->UploadDataIfDirty(Graphic::CurrentMaterial->m_Shader->GetBinder().get());
	}
}