module;
#include <string>
#include <vector>
export module RHI.IShaderReflection.VK;
import RHI.IShaderReflection;
import RHI.IEnum;

namespace SIByL::RHI
{
	export struct IShaderReflectionVK :public IShaderReflection
	{
	public:
		IShaderReflectionVK(char const* code, size_t size, ShaderStageFlagBits stage);
	};
}