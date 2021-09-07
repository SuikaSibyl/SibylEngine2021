#pragma once

namespace SIByL
{
	class TriangleMesh;
	class DrawItem;

	struct MeshFilterComponent
	{
		MeshFilterComponent(Ref<TriangleMesh> mesh);
		Ref<TriangleMesh> Mesh;
		Ref<DrawItem> DItem;
	};
}