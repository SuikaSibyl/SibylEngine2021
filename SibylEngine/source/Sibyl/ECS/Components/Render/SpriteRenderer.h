#pragma once

#include <glm/glm.hpp>

namespace SIByL
{
	class Texture2D;
	class Material;

	struct SpriteRendererComponent
	{
		Ref<Texture2D> Sprite = nullptr;
		Ref<Material> Material = nullptr;
		glm::vec4 Color{ 1.0f,1.0f,1.0f,1.0f };

		SpriteRendererComponent() = default;
		SpriteRendererComponent(glm::vec4 color, Ref<Texture2D> sprite = nullptr)
			:Color(color), Sprite(sprite) {}
		SpriteRendererComponent(const SpriteRendererComponent&) = default;
		SpriteRendererComponent(const glm::vec4& color) :Color(color) {}
	};
}