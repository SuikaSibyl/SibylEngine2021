#pragma once

#include <glm/glm.hpp>

namespace SIByL
{
	class Camera
	{
	public:

		const glm::vec3& GetPosition() const { return m_Posistion; }
		void SetPosition(const glm::vec3& position) { m_Posistion = position; }

		const glm::mat4& GetProjectionMatrix() const { return m_Projection; }
		const glm::mat4& GetViewMatrix() const { return m_View; }
		const glm::mat4& GetViewProjectionMatrix() const { return m_ViewProjection; }

	private:
		virtual void RecalculateViewMatrix() = 0;

	private:
		glm::vec3 m_Posistion;
		glm::vec3 m_Direction;
		glm::vec3 m_Up;
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

	};

}