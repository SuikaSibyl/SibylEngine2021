module;
#include <string>
#include <slang.h>
#include <slang-com-ptr.h>
export module RHI.IShader;
import Core.SObject;
import RHI.IEnum;
import RHI.IShaderReflection;

namespace SIByL
{
	namespace RHI
	{
		// A Shader tends to be a handle to a compiled blob of shader (HLSL, GLSL, MSL, etc.) code to be fed to a given Pipeline.
		// ╭──────────────┬────────────────────╮
		// │  Vulkan	  │   vk::ShaderModul  │
		// │  DirectX 12  │   ID3DBlob         │
		// │  OpenGL      │   GLuint           │
		// ╰──────────────┴────────────────────╯
		export struct ShaderDesc
		{
			ShaderStage stage;
			std::string entryPoint;
		};


		export class IShader : public SObject
		{
		public:
		public:
			IShader() = default;
			IShader(IShader&&) = default;
			virtual ~IShader() = default;
			virtual auto injectDesc(ShaderDesc const& desc) noexcept -> void = 0;
			virtual auto getReflection() noexcept -> IShaderReflection* = 0;
		};
	}
}