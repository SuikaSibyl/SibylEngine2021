export module Asset.Shader;
import Core.MemoryManager;
import Asset.Asset;
import RHI.IEnum;
import RHI.IShader;

namespace SIByL::Asset
{
	export struct Shader :public Asset
	{
		RHI::ShaderStage stage;
		MemScope<RHI::IShader> shader;
	};
}