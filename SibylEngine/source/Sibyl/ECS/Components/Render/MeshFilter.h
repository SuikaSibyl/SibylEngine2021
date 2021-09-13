#pragma once

namespace SIByL
{
	class TriangleMesh;
	class DrawItem;
	class ShaderConstantsBuffer;

	class PerObjectConstantsBufferPool
	{
	public:
		static Ref<ShaderConstantsBuffer> GetPerObjectConstantsBuffer();
		static void PushToPool(Ref<ShaderConstantsBuffer>);
	private:
		static std::vector<Ref<ShaderConstantsBuffer>> m_IdleConstantsBuffer;
		static std::vector<Ref<ShaderConstantsBuffer>> m_InFlightConstantsBuffer;
	};

	struct MeshFilterComponent
	{
		MeshFilterComponent();
		MeshFilterComponent(const MeshFilterComponent&) = default;
		~MeshFilterComponent();

		UINT GetSubmeshNum();
		Ref<TriangleMesh> Mesh;
		Ref<ShaderConstantsBuffer> PerObjectBuffer;
	};
}