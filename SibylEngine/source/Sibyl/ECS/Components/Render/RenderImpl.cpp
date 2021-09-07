#include "SIByLpch.h"
#include "MeshFilter.h"

#include "Sibyl/Graphic/Core/Geometry/TriangleMesh.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/DrawItem.h"

namespace SIByL
{
	MeshFilterComponent::MeshFilterComponent(Ref<TriangleMesh> mesh)
		:Mesh(mesh)
	{
		DItem = CreateRef<DrawItem>(mesh);
	}

}