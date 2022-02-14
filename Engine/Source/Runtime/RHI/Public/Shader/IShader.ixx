module;
#include <string>
#include <slang.h>
#include <slang-com-ptr.h>
export module RHI.IShader;
import Core.SObject;
import RHI.IEnum;

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

		export class IShader : public SObject
		{
		public:
		public:
			IShader() = default;
			IShader(IShader&&) = default;
			virtual ~IShader() = default;

		protected:
			ShaderStage stage;
		};
	}
}