#pragma once

#include "Sibyl/ECS/Components/Common/Transform.h"
#include "Sibyl/Basic/Random/LowDiscrepancySeq/LowDiscrepancySequence.h"

namespace SIByL
{
	class ScriptableCullingParameters;
	class ShaderConstantsBuffer;

	class Camera
	{
	public:
		Camera();

		virtual ~Camera() = default;
		Ref<TransformComponent> GetTransform() { return m_Transform; }
		void SetTransform(Ref<TransformComponent> transform)
		{
			SetPosition(transform->Translation);
			SetRotation(transform->EulerAngles);
		}

		const glm::vec3& GetPosition() const { return m_Posistion; }
		void SetPosition(const glm::vec3& position) {
			m_Transform->Translation = position;
			m_Posistion = position;
			RecalculateViewMatrix();
		}

		glm::vec3& MoveForward(const glm::vec3& offset) {
			glm::vec3 res = { 0,0,0 };
			res += m_Right * offset.x;
			res += m_Front * offset.z;
			res += m_Up * offset.y;
			return res;
		}

		const glm::vec3& GetRotation() const { return m_Rotation; }
		void SetRotation(const glm::vec3& rotation) { 
			m_Rotation = rotation;
			m_Transform->EulerAngles = rotation;
			m_Front.x = sin(glm::radians(rotation.y)) * cos(glm::radians(rotation.x));
			m_Front.y = sin(glm::radians(rotation.x));
			m_Front.z = cos(glm::radians(rotation.y)) * cos(glm::radians(rotation.x));
			m_Front = glm::normalize(m_Front);
			// also re-calculate the Right and Up vector
			// normalize the vectors, because their length gets closer to 0 
			// the more you look up or down which results in slower movement.
			if (m_Rotation.x < -89) m_Rotation.x = -89;
			else if (m_Rotation.x > 89) m_Rotation.x = 89;
			m_Right = glm::normalize(glm::cross(m_WorldUp, m_Front));
			m_Up = glm::normalize(glm::cross(m_Front, m_Right));
			RecalculateViewMatrix(); 
		}

		virtual void Dither(double x, double y) {}
		virtual void RecalculateProjectionMatrix() = 0;
		void Resize(float width, float height)
		{
			if (width <= 0 || height <= 0)
				return;

			m_Width = width;
			m_Height = height;
			RecalculateProjectionMatrix();
		}

		const glm::mat4& GetProjectionMatrix() const { return m_Projection; }
		const glm::mat4& GetViewMatrix() const { return m_View; }
		const glm::mat4& GetProjectionViewMatrix() const { return m_ProjectionView; }

		virtual bool TryGetCullingParameters(ScriptableCullingParameters& p) { return true; }

	protected:
		virtual void RecalculateViewMatrix()
		{
			glm::vec3 center = m_Posistion + m_Front;
			m_View = glm::lookAtLH(m_Posistion, center, m_Up);
			m_ProjectionView = m_Projection * m_View;

			UpdateViewConstant();
		}

	protected:
		glm::vec3 m_Posistion;
		glm::vec3 m_Direction;
		glm::vec3 m_Rotation;

		glm::vec3 m_Front;
		glm::vec3 m_Up;
		glm::vec3 m_WorldUp;
		glm::vec3 m_Right;

		glm::mat4 m_Projection;
		glm::mat4 m_View;
		glm::mat4 m_ProjectionView;
		glm::mat4 m_PreviousProjectionView;
		glm::mat4 m_CurrentProjectionView;

		Ref<TransformComponent> m_Transform;

		float m_Width;
		float m_Height;
		bool m_ApplyDither = false;
	public:
		void SetCamera();
		void RecordVPMatrix();
		void SetApplyDither(bool dither) { m_ApplyDither = dither; }
		void OnDrawCall();
		ShaderConstantsBuffer& GetConstantsBuffer();
	protected:
		void UpdateProjectionConstant();
		void UpdatePreviousViewProjectionConstant();
		void UpdateViewConstant();
		virtual glm::mat4 GetPreciseProjectionMatrix() = 0;
		Ref<ShaderConstantsBuffer> m_ConstantsBuffer = nullptr;
	};

	class OrthographicCamera :public Camera
	{
	public:
		OrthographicCamera(float width, float height)
			:Camera()
		{
			m_Width = width;
			m_Height = height;
			RecalculateProjectionMatrix();
		}

	protected:
		virtual glm::mat4 GetPreciseProjectionMatrix() override
		{
			return glm::orthoLH_NO(-1.0f, 1.0f, -1.0f, 1.0f, -100.0f, 100.0f);
		}
		virtual void RecalculateProjectionMatrix() override
		{
			m_Projection = glm::orthoLH_NO(-1.0f, 1.0f, -1.0f, 1.0f, -100.0f, 100.0f);
			UpdateProjectionConstant();
			m_ProjectionView = m_Projection * m_View;
		}
	};

	class PerspectiveCamera :public Camera
	{
	public:
		PerspectiveCamera(float fov, float width, float height)
			:m_FoV(fov), Camera()
		{
			m_Width = width;
			m_Height = height;
			RecalculateProjectionMatrix();
		}

	protected:
		virtual void RecalculateProjectionMatrix() override
		{
			m_Projection = glm::perspectiveLH_NO(glm::radians(m_FoV), m_Width / m_Height, 0.001f, 100.0f);
			UpdateProjectionConstant();
		}

		virtual glm::mat4 GetPreciseProjectionMatrix() override
		{
			return glm::perspectiveLH_NO(glm::radians(m_FoV), m_Width / m_Height, 0.001f, 100.0f);
		}

		virtual void Dither(double x, double y) override 
		{
			m_Projection = glm::perspectiveLH_NO(glm::radians(m_FoV), m_Width / m_Height, 0.001f, 100.0f);
			m_Projection[2][0] += (x * 2 - 1) / m_Width;
			m_Projection[2][1] += (y * 2 - 1) / m_Height;
			UpdateProjectionConstant();
			m_ProjectionView = m_Projection * m_View;
		}

	private:
		float m_FoV;
	};

}