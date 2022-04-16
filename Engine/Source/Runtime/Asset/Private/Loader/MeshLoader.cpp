module;
#include <string>
#include <filesystem>
#include <glm/glm.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
module Asset.MeshLoader;
import Core.Log;
import Core.Buffer;
import Core.Cache;
import Core.MemoryManager;
import Asset.Asset;
import Asset.DedicatedLoader;
import Asset.Mesh;
import RHI.IFactory;
import RHI.IVertexBuffer;
import RHI.IIndexBuffer;

namespace SIByL::Asset
{
	struct Vertex {
		glm::vec3 Position;
		glm::vec3 Normal;
		glm::vec2 TexCoords;
	};

	auto processMesh(aiMesh* mesh, const aiScene* scene) noexcept -> void
	{
		//vector<Vertex> vertices;
		//vector<unsigned int> indices;
		//vector<Texture> textures;

		//for (unsigned int i = 0; i < mesh->mNumVertices; i++)
		//{
		//	Vertex vertex;
		//	glm::vec3 vector;
		//	vector.x = mesh->mVertices[i].x;
		//	vector.y = mesh->mVertices[i].y;
		//	vector.z = mesh->mVertices[i].z;
		//	vertex.Position = vector;

		//	vector.x = mesh->mNormals[i].x;
		//	vector.y = mesh->mNormals[i].y;
		//	vector.z = mesh->mNormals[i].z;
		//	vertex.Normal = vector;

		//}
		//// 处理索引
		//for (unsigned int i = 0; i < mesh->mNumFaces; i++)
		//{
		//	aiFace face = mesh->mFaces[i];
		//	for (unsigned int j = 0; j < face.mNumIndices; j++)
		//		indices.push_back(face.mIndices[j]);
		//}

		//return Mesh(vertices, indices, textures);
	}

	auto processNode(aiNode* node, const aiScene* scene) noexcept -> void
	{
		// process every mesh
		for (unsigned int i = 0; i < node->mNumMeshes; i++)
		{
			aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
			//meshes.push_back(processMesh(mesh, scene));
		}
		// repeat for every children
		for (unsigned int i = 0; i < node->mNumChildren; i++)
		{
			processNode(node->mChildren[i], scene);
		}
	}

	auto MeshLoader::loadFromFile(std::filesystem::path path) noexcept -> void
	{
		Assimp::Importer importer;
		const aiScene* scene = importer.ReadFile(path.string(), aiProcess_Triangulate | aiProcess_FlipUVs);

		if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
		{
			SE_CORE_ERROR("Asset :: ASSIMP :: {0}", importer.GetErrorString());
			return;
		}
		processNode(scene->mRootNode, scene);
	}
}