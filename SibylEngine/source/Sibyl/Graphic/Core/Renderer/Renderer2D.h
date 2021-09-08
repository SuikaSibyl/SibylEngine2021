#pragma once

#include "Sibyl/Graphic/AbstractAPI/Core/Top/Camera.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/Texture.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Material.h"

namespace SIByL
{
	class Renderer2D
	{
	public:
		static void Init();
		static void Shutdown();

		static void BeginScene(Ref<Camera> camera);
		static void EndScene();

		static Ref<Material> GetMaterial();

		// Primitives
		static void DrawQuad(const glm::mat4& transform, const glm::vec4& color);
		static void DrawQuad(const glm::mat4& transform, const glm::vec4& color, Ref<Texture2D> texture);

		static void DrawQuad(const glm::vec2& position, const glm::vec2& size, const glm::vec4& color);
		static void DrawQuad(const glm::vec3& position, const glm::vec2& size, const glm::vec4& color);
		static void DrawQuad(const glm::vec2& position, const glm::vec2& size, const glm::vec4& color, Ref<Texture2D> texture);
		static void DrawQuad(const glm::vec3& position, const glm::vec2& size, const glm::vec4& color, Ref<Texture2D> texture);
	};
}