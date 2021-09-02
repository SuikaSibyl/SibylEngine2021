#pragma once

#include "Sibyl/Components/Common/Transform.h"

namespace SIByL
{
	class Camera
	{
	public:
		Camera()
		{
			m_Transform = std::make_shared<Transform>();
			m_WorldUp = m_Up = glm::vec3(0.0f, 1.0f, 0.0f);
			SetPosition(glm::vec3(0.0f, 0.0f, -1.0f));
			SetRotation(glm::vec3(0.0f, 0.0f, 0.0f));
		}

		Ref<Transform> GetTransform() { return m_Transform; }
		void SetTransform(Ref<Transform> transform)
		{
			SetPosition(transform->position);
			SetRotation(transform->eulerAngles);
		}

		const glm::vec3& GetPosition() const { return m_Posistion; }
		void SetPosition(const glm::vec3& position) {
			m_Transform->position = position;
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
			m_Transform->eulerAngles = rotation;
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
		const glm::mat4& GetViewProjectionMatrix() const { return m_ViewProjection; }

	protected:
		virtual void RecalculateViewMatrix()
		{
			glm::vec3 center = m_Posistion + m_Front;
			m_View = glm::lookAtLH(m_Posistion, center, m_Up);
			m_ViewProjection = m_View * m_Projection;
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
		glm::mat4 m_ViewProjection;

		Ref<Transform> m_Transform;

		float m_Width;
		float m_Height;
	};

	class OrthographicCamera :public Camera
	{
	public:
		OrthographicCamera(float width, float height)
		{
			m_Width = width;
			m_Height = height;
			RecalculateProjectionMatrix();
		}

	protected:
		virtual void RecalculateProjectionMatrix() override
		{
			m_Projection = glm::orthoLH_NO(-1.0f, 1.0f, -1.0f, 1.0f, -100.0f, 100.0f);
		}
	};

	class PerspectiveCamera :public Camera
	{
	public:
		PerspectiveCamera(float fov, float width, float height)
			:m_FoV(fov)
		{
			m_Width = width;
			m_Height = height;
			RecalculateProjectionMatrix();
		}

	protected:
		virtual void RecalculateProjectionMatrix() override
		{
			m_Projection = glm::perspectiveLH_NO(glm::radians(m_FoV), m_Width / m_Height, 0.001f, 100.0f);
		}

	private:
		float m_FoV;
	};

}