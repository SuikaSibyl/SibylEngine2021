#include "CudaModulePCH.h"
#include "CuBVHInterface.h"

#include "CuBVH.h"

namespace SIByL
{
	ICuBVH::ICuBVH()
	{
		p_CuBVH = new CuBVH();
	}

	ICuBVH::~ICuBVH()
	{
		delete p_CuBVH;
	}

	void ICuBVH::LoadData(const std::vector<float>& vertices, const std::vector<uint32_t>& indices, unsigned int vnum)
	{
		p_CuBVH->LoadData(vertices, indices, vertices.size() / vnum);
	}

}