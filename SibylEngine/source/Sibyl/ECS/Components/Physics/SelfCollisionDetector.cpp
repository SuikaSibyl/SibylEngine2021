#include "SIByLpch.h"
#include "SelfCollisionDetector.h"

#include "Sibyl/ECS/Components/Components.h"
#include "Sibyl/Graphic/Core/Geometry/MeshLoader.h"
#include "Sibyl/Graphic/Core/Geometry/ObjLoader.h"
#include <Sibyl/Graphic/AbstractAPI/Core/Middle/VertexBuffer.h>

#ifdef SIBYL_PLATFORM_CUDA
#include "CudaModule/source/CudaModule.h"
#endif

namespace SIByL
{
	void SelfCollisionDetectorComponent::Init(MeshFilterComponent& mf)
	{
		std::cout << "Self Collision Detector Start" << std::endl;

#ifdef SIBYL_PLATFORM_CUDA
		// Load Data from File
		SIByL::VertexBufferLayout SimpleLayout =
		{
			{SIByL::ShaderDataType::Float3, "POSITION"},
		};

		ObjLoader objLoader(mf.Path);
		//MeshLoader meshLoader(mf.Path, SimpleLayout);
		SMeshCacheAsset* meshCache = &objLoader.m_MeshCacheAsset;

		std::vector<std::pair<int, int>> collided_pairs;
		ICollisionDetection::SelfCollisionDetect(meshCache->m_Meshes[0].vertices, meshCache->m_Meshes[0].indices, collided_pairs);
		//ICuBVH bvh;
		//bvh.LoadData(meshCache->m_Meshes[0].vertices, meshCache->m_Meshes[0].indices, meshCache->m_Meshes[0].vertices.size() / 3);
		for (auto& pair : collided_pairs)
		{
			std::cout << "pair detected: " << pair.first << ", " << pair.second << std::endl;
		}

#endif
	}
}