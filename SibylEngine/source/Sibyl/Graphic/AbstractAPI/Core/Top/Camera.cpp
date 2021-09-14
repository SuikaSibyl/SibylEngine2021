#include "SIByLpch.h"
#include "Camera.h"

#include "Graphic.h"

#include <Sibyl/Graphic/AbstractAPI/Core/Middle/Shader.h>
#include <Sibyl/Graphic/AbstractAPI/Core/Middle/ShaderBinder.h>
#include <Sibyl/Graphic/AbstractAPI/Core/Top/Material.h>

namespace SIByL
{
	static ShaderConstantsDesc PerCameraConstantsDesc;
	static ShaderConstantsDesc* GetPerCameraConstantsDesc()
	{
		// Init if not Inited
		if (PerCameraConstantsDesc.Size == -1)
		{
			ConstantBufferLayout& layout = ConstantBufferLayout::PerCameraConstants;
			int paraIndex = 0;

			PerCameraConstantsDesc.Size = layout.GetStide();
			for (auto bufferElement : layout)
			{
				PerCameraConstantsDesc.Mapper.InsertConstant(bufferElement, 2);
			}
		}

		return &PerCameraConstantsDesc;
	}


	Camera::Camera()
	{
		m_Transform = std::make_shared<TransformComponent>();
		m_WorldUp = m_Up = glm::vec3(0.0f, 1.0f, 0.0f);
		SetPosition(glm::vec3(0.0f, 0.0f, -1.0f));
		SetRotation(glm::vec3(0.0f, 0.0f, 0.0f));

		ShaderConstantsDesc* desc = GetPerCameraConstantsDesc();
		m_ConstantsBuffer = ShaderConstantsBuffer::Create(desc);
	}

	void Camera::SetCamera()
	{
		// Current Camera
		Graphic::CurrentCamera = this;
	}

	void Camera::OnDrawCall()
	{
		// Upload Per-Material parameters to GPU
		m_ConstantsBuffer->UploadDataIfDirty(Graphic::CurrentMaterial->m_Shader->GetBinder().get());
	}

	ShaderConstantsBuffer& Camera::GetConstantsBuffer()
	{
		return *m_ConstantsBuffer;
	}

	void Camera::UpdateProjectionConstant()
	{
		if (m_ConstantsBuffer)
			m_ConstantsBuffer->SetMatrix4x4("Projection", m_Projection);
	}

	void Camera::UpdateViewConstant()
	{
		if (m_ConstantsBuffer)
			m_ConstantsBuffer->SetMatrix4x4("View", m_View);
	}
}