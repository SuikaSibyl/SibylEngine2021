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
import Math.LinearAlgebra;

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
		std::filesystem::path fullpath = "./assets/" / path;
		const aiScene* scene = importer.ReadFile(fullpath.string(), aiProcess_Triangulate | aiProcess_FlipUVs);

		if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
		{
			SE_CORE_ERROR("Asset :: ASSIMP :: {0}", importer.GetErrorString());
			return;
		}
		processNode(scene->mRootNode, scene);
	}

	auto ExternalMeshSniffer::loadFromFile(std::filesystem::path path) noexcept -> Node
	{
		std::filesystem::path fullpath = "./assets/" / path;
		scene = importer.ReadFile(fullpath.string(), aiProcess_Triangulate | aiProcess_FlipUVs);

		if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
		{
			SE_CORE_ERROR("Asset :: ASSIMP :: {0}", importer.GetErrorString());
			return (void*)0;
		}
		return (void*)scene->mRootNode;
	}

	auto ExternalMeshSniffer::interpretNode(Node node, uint32_t& mesh_num, uint32_t& children_num, std::string& name) noexcept -> void
	{
		aiNode* ainode = (aiNode*)node;
		mesh_num = ainode->mNumMeshes;
		children_num = ainode->mNumChildren;
		name = ainode->mName.C_Str();
	}

	auto ExternalMeshSniffer::getNodeChildren(Node node, uint32_t index) noexcept -> Node
	{
		return (Node)(((aiNode*)node)->mChildren[index]);
	}

	auto ExternalMeshSniffer::fillVertexIndex(Node node, Buffer& vb, Buffer& ib) noexcept -> void
	{
		aiNode* ainode = (aiNode*)node;
		aiMesh* mesh = scene->mMeshes[ainode->mMeshes[0]];

		// Load Vertices
		uint64_t vertices_count = mesh->mNumVertices;
		vb = std::move(Buffer(vertices_count * 8 * 12, 8 * 12));
		float* vertices = (float*)vb.getData();
		uint64_t vidx = 0;
		for (unsigned int i = 0; i < mesh->mNumVertices; i++)
		{
			vertices[vidx++] = mesh->mVertices[i].x;
			vertices[vidx++] = mesh->mVertices[i].y;
			vertices[vidx++] = mesh->mVertices[i].z;

			vertices[vidx++] = mesh->mNormals[i].x;
			vertices[vidx++] = mesh->mNormals[i].y;
			vertices[vidx++] = mesh->mNormals[i].z;

			if (mesh->HasTextureCoords(0))
			{
				vertices[vidx++] = (mesh->mTextureCoords[1][i].x);
				vertices[vidx++] = (mesh->mTextureCoords[1][i].y);
			}
			else
			{
				vertices[vidx++] = (0);
				vertices[vidx++] = (0);
			}

			if (mesh->HasTangentsAndBitangents())
			{
				vertices[vidx++] = (mesh->mTangents[i].x);
				vertices[vidx++] = (mesh->mTangents[i].y);
				vertices[vidx++] = (mesh->mTangents[i].z);

				glm::vec3 Normal(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
				glm::vec3 Tangent(mesh->mTangents[i].x, mesh->mTangents[i].y, mesh->mTangents[i].z);
				glm::vec3 Bitangent = glm::cross(Normal, Tangent);
				if (Bitangent.x * mesh->mBitangents[i].x)
					vertices[vidx++] = (1);
				else
					vertices[vidx++] = (0);
			}
			else
			{
				//SIByL_CORE_ERROR("Mesh Process Error: NO TANGENT");
				vertices[vidx++] = (0);
				vertices[vidx++] = (0);
				vertices[vidx++] = (0);
				vertices[vidx++] = (0);
			}
		}


		// Load Indices
		uint64_t indices_count = mesh->mNumFaces * 3;
		uint32_t indices_width = (indices_count > 65535) ? 32 : 16;
		bool exceed32 = (indices_count > 4294967295);

		if (exceed32)
		{
			SE_CORE_ERROR("Asset :: Assimp :: Too Much Indices to handle by uint32_t");
		}

		ib = std::move(Buffer(indices_count * indices_width / 8, indices_width / 8));
		if (indices_width == 32)
		{
			uint32_t* indices = (uint32_t*)ib.getData();
			uint32_t idx = 0;
			for (unsigned int i = 0; i < mesh->mNumFaces; i++)
			{
				aiFace face = mesh->mFaces[i];
				for (unsigned int j = 0; j < face.mNumIndices; j++)
					indices[idx++] = (face.mIndices[j]);
			}
		}
		else
		{
			uint16_t* indices = (uint16_t*)ib.getData();
			uint16_t idx = 0;
			for (unsigned int i = 0; i < mesh->mNumFaces; i++)
			{
				aiFace face = mesh->mFaces[i];
				for (unsigned int j = 0; j < face.mNumIndices; j++)
					indices[idx++] = (face.mIndices[j]);
			}
		}
	}

}