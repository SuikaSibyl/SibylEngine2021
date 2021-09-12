#pragma once

class aiNode;
class aiScene;
class aiMesh;

#include <Sibyl/Graphic/AbstractAPI/Core/Middle/VertexBuffer.h>

namespace SIByL
{
	class TriangleMesh;

	struct MeshData
	{
		std::vector<float> vertices;
		std::vector<uint32_t> indices;

		uint32_t vNum;
		uint32_t iNum;

		MeshData(const std::vector<float>& v,
			const std::vector<uint32_t>& i,
			const uint32_t& vn, const uint32_t& in)
			:vertices(v), indices(i), vNum(vn), iNum(in) {}
	};

	class MeshLoader
	{
	public:
		MeshLoader(const std::string& path, const VertexBufferLayout& layout);
		Ref<TriangleMesh> GetTriangleMesh();

	private:
		void LoadFile(const std::string& path);
		void ProcessNode(aiNode* node, const aiScene* scene);
		MeshData ProcessMesh(aiMesh* mesh, const aiScene* scene);
		VertexBufferLayout m_Layout;
		std::vector<MeshData> m_Meshes;
		std::string m_Directory;
	};
}