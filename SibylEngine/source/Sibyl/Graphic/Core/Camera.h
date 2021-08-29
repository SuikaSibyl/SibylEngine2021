#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace SIByL
{
	class Camera
	{
	public:
		Camera()
		{
			m_WorldUp = m_Up = glm::vec3(0.0f, 1.0f, 0.0f);
			SetPosition(glm::vec3(0.0f, 0.0f, -1.0f));
			SetRotation(glm::vec3(0.0f, 0.0f, 0.0f));
		}

		const glm::vec3& GetPosition() const { return m_Posistion; }
		void SetPosition(const glm::vec3& position) {
			m_Posistion = position;
			RecalculateViewMatrix();
		}

		const glm::vec3& GetRotation() const { return m_Rotation; }
		void SetRotation(const glm::vec3& rotation) { 
			m_Rotation = rotation;
			m_Front.x = sin(glm::radians(rotation.x)) * cos(glm::radians(rotation.y));
			m_Front.y = sin(glm::radians(rotation.y));
			m_Front.z = cos(glm::radians(rotation.x)) * cos(glm::radians(rotation.y));
			m_Front = glm::normalize(m_Front);
			// also re-calculate the Right and Up vector
			// normalize the vectors, because their length gets closer to 0 
			// the more you look up or down which results in slower movement.
			m_Right = glm::normalize(glm::cross(m_WorldUp, m_Front));
			m_Up = glm::normalize(glm::cross(m_Front, m_Right));
			RecalculateViewMatrix(); 
		}

		const glm::mat4& GetProjectionMatrix() const { return m_Projection; }
		const glm::mat4& GetViewMatrix() const { return m_View; }
		const glm::mat4& GetViewProjectionMatrix() const { return m_ViewProjection; }

	protected:
		virtual void RecalculateViewMatrix()
		{
			m_View = glm::lookAt(m_Posistion, m_Posistion + m_Front, m_Up);
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
	};

	class OrthographicCamera :public Camera
	{

	};

	class PerspectiveCamera :public Camera
	{
	public:
		PerspectiveCamera(float fov, float width, float height)
			:m_FoV(fov), m_Width(width), m_Height(height)
		{
			RecalculateProjectionMatrix();
		}

	protected:
		void RecalculateProjectionMatrix()
		{
			m_Projection = glm::perspective(glm::radians(m_FoV), m_Width / m_Height, 0.1f, 100.0f);
		}

	private:
		float m_FoV;
		float m_Width;
		float m_Height;
	};

}