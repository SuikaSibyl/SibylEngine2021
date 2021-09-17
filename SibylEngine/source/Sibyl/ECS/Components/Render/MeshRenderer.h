#pragma once

namespace SIByL
{
	class TriangleMesh;
	class Texture2D;
	class Material;

	struct MeshRendererComponent
	{
		std::vector<Ref<Material>> Materials;

		UINT MaterialNum;

		MeshRendererComponent() = default;
		MeshRendererComponent(const MeshRendererComponent&) = default;
		void SetMaterialNums(int num);
	};
}