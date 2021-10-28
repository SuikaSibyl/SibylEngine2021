#pragma once

#include "MeshLoader.h"

namespace SIByL
{
	class ObjLoader
	{
	public:
		ObjLoader(std::string filePath);
		SMeshCacheAsset m_MeshCacheAsset;
	};
}