#pragma once

#include <glm/glm.hpp>
#include "Sibyl/ECS/Asset/CustomAsset.h"

namespace SIByL
{
	class ComputeShader;
	class ShaderConstantsBuffer;
	class ShaderResourcesBuffer;
	class UnorderedAccessBuffer;
	class ShaderConstantsDesc;
	class ShaderResourcesDesc;
	class RenderTarget;
	class Texture2D;
	class FrameBuffer;

	class ComputeInstance :public CustomAsset
	{
	public:
		void Dispatch(unsigned int x, unsigned int y, unsigned int z);

		////////////////////////////////////////////////////////////////////
		///					Parameter Setter / Getter					 ///
		void SetFloat(const std::string& name, const float& value);
		void SetFloat2(const std::string& name, const glm::vec2& value);
		void SetFloat3(const std::string& name, const glm::vec3& value);
		void SetFloat4(const std::string& name, const glm::vec4& value);
		void SetMatrix4x4(const std::string& name, const glm::mat4& value);
		void SetTexture2D(const std::string& name, Ref<Texture2D> texture);
		void SetTexture2D(const std::string& name, RenderTarget* texture);
		void SetRenderTarget2D(const std::string& name, Ref<FrameBuffer> framebuffer, unsigned int attachmentIdx);

		////////////////////////////////////////////////////////////////////
		///							Initializer							 ///
		ComputeInstance() = default;
		ComputeInstance(Ref<ComputeShader> shader);
		void UseShader(Ref<ComputeShader> shader);
		void OnDrawCall();

		////////////////////////////////////////////////////////////////////
		///							Serializer							 ///
		virtual void SaveAsset() override;

	private:
		Ref<ComputeShader> m_Shader = nullptr;
		Ref<ShaderConstantsBuffer> m_ConstantsBuffer = nullptr;
		Ref<ShaderResourcesBuffer> m_ResourcesBuffer = nullptr;
		Ref<UnorderedAccessBuffer> m_UnorderedAccessBuffer = nullptr;

		ShaderConstantsDesc* m_ConstantsDesc = nullptr;
		ShaderResourcesDesc* m_ResourcesDesc = nullptr;
		ShaderResourcesDesc* m_UnorderedAccessDesc = nullptr;
	};
}