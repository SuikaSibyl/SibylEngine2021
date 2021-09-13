#include "SIByLpch.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "MeshLoader.h"
#include "Sibyl/Graphic/Core/Geometry/TriangleMesh.h"
#include "Sibyl/Graphic/Core/Renderer/Renderer.h"

namespace SIByL
{
    MeshLoader::MeshLoader(const std::string& path, const VertexBufferLayout& layout)
        :m_Layout(layout)
    {
        LoadFile("../Assets/" + path);
    }

	void MeshLoader::LoadFile(const std::string& path)
	{
		Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(path, 
            ((Renderer::GetRaster() == RasterRenderer::OpenGL) ? aiProcess_FlipUVs : 0) | 
            aiProcess_Triangulate | aiProcess_GenNormals);

        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
        {
            SIByL_CORE_ERROR("Mesh Load Error");
            return;
        }

        m_Directory = path.substr(0, path.find_last_of('/'));

        ProcessNode(scene->mRootNode, scene);
	}

    void MeshLoader::ProcessNode(aiNode* node, const aiScene* scene)
    {
        for (unsigned int i = 0; i < node->mNumMeshes; i++)
        {
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            m_Meshes.push_back(ProcessMesh(mesh, scene));
        }
        // Iteratorly Process Child Node
        for (unsigned int i = 0; i < node->mNumChildren; i++)
        {
            ProcessNode(node->mChildren[i], scene);
        }
    }

    MeshData MeshLoader::ProcessMesh(aiMesh* mesh, const aiScene* scene)
    {
        std::vector<float> vertices;
        std::vector<uint32_t> indices;

        for (unsigned int i = 0; i < mesh->mNumVertices; i++)
        {
            for (BufferElement& element : m_Layout)
            {
                if (element.Name == "POSITION")
                {
                    vertices.push_back(mesh->mVertices[i].x);
                    vertices.push_back(mesh->mVertices[i].y);
                    vertices.push_back(mesh->mVertices[i].z);
                }
                else if (element.Name == "TEXCOORD" || element.Name == "TEXCO0RD0")
                {
                    vertices.push_back(mesh->mTextureCoords[0][i].x);
                    vertices.push_back(mesh->mTextureCoords[0][i].y);
                }
                else if (element.Name == "NORMAL")
                {
                    vertices.push_back(mesh->mNormals[i].x);
                    vertices.push_back(mesh->mNormals[i].y);
                    vertices.push_back(mesh->mNormals[i].z);
                }
                else
                {
                    for (int j = 0; j < element.Size / 4; j++)
                        vertices.push_back(0);
                }
            }
        }

        for (unsigned int i = 0; i < mesh->mNumFaces; i++)
        {
            aiFace face = mesh->mFaces[i];
            for (unsigned int j = 0; j < face.mNumIndices; j++)
                indices.push_back(face.mIndices[j]);
        }

        return MeshData(vertices, indices, mesh->mNumVertices, mesh->mNumFaces * 3);
    }

    Ref<TriangleMesh> MeshLoader::GetTriangleMesh()
    {
        return TriangleMesh::Create(m_Meshes, m_Layout);
    }
}