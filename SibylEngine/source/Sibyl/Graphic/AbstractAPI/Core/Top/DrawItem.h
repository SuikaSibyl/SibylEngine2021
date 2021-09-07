#pragma once

#include "glm/glm.hpp"

namespace SIByL
{
	class TriangleMesh;
	class ShaderConstantsBuffer;

	class PerObjectConstantsBufferPool
	{
	public:
		static Ref<ShaderConstantsBuffer> GetPerObjectConstantsBuffer();
		static void PushToPool(Ref<ShaderConstantsBuffer>);
	private:
		static std::vector<Ref<ShaderConstantsBuffer>> m_IdleConstantsBuffer;
	};

	class DrawItem
	{
	public:
		DrawItem(Ref<TriangleMesh> mesh);
		~DrawItem();

		void SetObjectMatrix(const glm::mat4& transform);
		void OnDrawCall();

	private:
		Ref<TriangleMesh> m_Mesh;
		Ref<ShaderConstantsBuffer> m_ConstantsBuffer = nullptr;
	};
}