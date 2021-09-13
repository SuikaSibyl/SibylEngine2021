#pragma once

namespace SIByL
{
	class TriangleMesh;
	class Texture2D;
	class Material;

	struct MeshRendererComponent
	{
		Ref<TriangleMesh> Mesh;
		std::vector<Ref<Material>> Materials;

		UINT MaterialNum;

		MeshRendererComponent() = default;
		MeshRendererComponent(const MeshRendererComponent&) = default;
	};
}