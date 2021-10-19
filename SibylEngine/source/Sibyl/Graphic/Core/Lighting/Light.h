#pragma once

#include <glm/glm.hpp>

namespace SIByL
{
	enum class LightType
	{
		None,
		Directional,
		Point,
	};

	class Light
	{
	public:
		virtual ~Light() = default;
		virtual void DrawImGui() = 0;
	private:

	};

	class DirectionalLight :public Light
	{
	public:

		virtual void DrawImGui() override;
		
		// Direction
		const glm::vec3 GetDirection() { return m_Direction; }
		void SetDirection(const glm::vec3& direction) { m_Direction = direction; }
		// Color
		const glm::vec3 GetColor() { return m_Color; }
		void SetColor(const glm::vec3& color) { m_Color = color; }

	protected:
		glm::vec3 m_Direction;
		glm::vec3 m_Color;
		float m_Intensity = 1;
	};

	class PointLight :public Light
	{
	public:
		virtual void DrawImGui() override;

	protected:
		glm::vec3 m_Position;
		glm::vec3 m_Color;
		float m_Intensity = 1;
	};
}