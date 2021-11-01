#pragma once

namespace SIByL
{
	class TriangleMesh;
	class Texture2D;
	class Material;

	struct MeshRendererComponent
	{
		std::vector<Ref<Material>>& GetPassMaterials(const std::string& passName);

		std::unordered_map<std::string, std::vector<Ref<Material>>> Materials;
		UINT SubmeshNum;

		MeshRendererComponent() = default;
		MeshRendererComponent(const MeshRendererComponent&) = default;
		void SetMaterialNums(int num);
	};
}