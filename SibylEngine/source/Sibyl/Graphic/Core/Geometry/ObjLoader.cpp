#include "SIByLpch.h"
#include "ObjLoader.h"

namespace SIByL
{
	ObjLoader::ObjLoader(std::string filePath)
		:m_MeshCacheAsset(filePath + ".simpleobj")
	{
		if (m_MeshCacheAsset.LoadCache())
		{
			return;
		}

		std::ifstream objfile("../Assets/" + filePath);
		SIByL_CORE_ASSERT(objfile.is_open(), "Obj file open failed!");
		
		MeshData meshData;
		std::string type;
		float v_texture0, v_texture1;
		float v_normalx, v_normaly, v_normalz;
		float v_posx, v_posy, v_posz;
		float f_td;
		char c;
		unsigned int f_vertex0, f_vertex1, f_vertex2;
		unsigned int f_tex0, f_tex1, f_tex2;
		unsigned int f_normal0, f_normal1, f_normal2;
		while (!objfile.eof())
		{
			objfile >> type;
			if (type == "vt")
			{
				objfile >> v_texture0 >> v_texture1;
			}
			else if (type == "ny" || type == "vn")
			{
				objfile >> v_normalx >> v_normaly >> v_normalz;
			}
			else if (type == "v")
			{
				objfile >> v_posx >> v_posy >> v_posz;
				meshData.vertices.push_back(v_posx);
				meshData.vertices.push_back(v_posy);
				meshData.vertices.push_back(v_posz);
			}
			else if (type == "f")
			{
				objfile >> f_vertex0 >> c >> f_normal0;// >> c >> f_tex0;
				objfile >> f_vertex1 >> c >> f_normal1;// >> c >> f_tex0;
				objfile >> f_vertex2 >> c >> f_normal2;// >> c >> f_tex0;
				meshData.indices.push_back(f_vertex0 - 1);
				meshData.indices.push_back(f_vertex1 - 1);
				meshData.indices.push_back(f_vertex2 - 1);
			}
			else if (type == "td")
			{
				objfile >> f_td;
			}
			else
			{
				std::cout << "?";
			}
			//cost[v][w] = weight;
			//cost[w][v] = weight;
		}
		meshData.vNum = meshData.vertices.size() / 3;
		meshData.iNum = meshData.indices.size();
		m_MeshCacheAsset.m_Meshes.emplace_back(meshData);
		m_MeshCacheAsset.SaveCache();
	}

}