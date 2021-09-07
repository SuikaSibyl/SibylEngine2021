#pragma once

namespace SIByL
{
	class RenderTarget;
	class Material;
	class Camera;
	class DrawItem;

	class Graphic
	{
	public:
		static void SetRenderTarget(const std::string& key);
		static void SetRenderTarget(Ref<RenderTarget> key);

		static void DrawDrawItemNow(Ref<DrawItem> drawItem);

	public:
		static Material* CurrentMaterial;
		static Camera* CurrentCamera;
	};
}