module;
#include <vector>
export module RHI.IFactory;
import Core.MemoryManager;
import Core.Buffer;
import RHI.IEnum;
import RHI.GraphicContext;
import RHI.IPhysicalDevice;
import RHI.ILogicalDevice;
import RHI.ISwapChain;
import RHI.IShader;
import RHI.IFixedFunctions;

namespace SIByL
{
	namespace RHI
	{
		export struct GraphicContextDesc
		{
			API api;
		};

		export struct PhysicalDeviceDesc
		{
			IGraphicContext* context;
		};

		export struct LogicalDeviceDesc
		{
			IPhysicalDevice* physicalDevice;
		};

		export struct SwapchainDesc
		{
			ILogicalDevice* logicalDevice;
		};

		export class IFactory
		{
		public:
			static auto createGraphicContext(GraphicContextDesc const& desc) noexcept -> IGraphicContext*;
			static auto createPhysicalDevice(PhysicalDeviceDesc const& desc) noexcept -> IPhysicalDevice*;
			static auto createLogicalDevice(LogicalDeviceDesc const& desc) noexcept -> ILogicalDevice*;
			static auto createSwapchain(SwapchainDesc const& desc) noexcept -> ISwapChain*;
		};

		export class IResourceFactory
		{
		public:
			IResourceFactory(ILogicalDevice* logical_device);

			auto createShaderFromBinary(Buffer const& binary, ShaderDesc const& desc) noexcept -> MemScope<IShader>;

			auto createVertexLayout() noexcept -> MemScope<IVertexLayout>;
			auto createInputAssembly(TopologyKind const& kind) noexcept -> MemScope<IInputAssembly>;
			auto createViewportsScissors(
				unsigned int width_viewport,
				unsigned int height_viewport,
				unsigned int width_scissor,
				unsigned int height_scissor) noexcept -> MemScope<IViewportsScissors>;
			auto createViewportsScissors(Extend const& extend_viewport, Extend const& extend_scissor) noexcept -> MemScope<IViewportsScissors>;
			auto createRasterizer(RasterizerDesc const& desc) noexcept -> MemScope<IRasterizer>;
			auto createMultisampling(MultiSampleDesc const& desc) noexcept -> MemScope<IMultisampling>;
			auto createDepthStencil(DepthStencilDesc const& desc) noexcept -> MemScope<IDepthStencil>;
			auto createColorBlending(ColorBlendingDesc const& desc) noexcept -> MemScope<IColorBlending>;
			auto createDynamicState(std::vector<PipelineState> const& states) noexcept -> MemScope<IDynamicState>;

		private:
			API api;
			IGraphicContext* graphicContext;
			IPhysicalDevice* physicalDevice;
			ILogicalDevice* logicalDevice;
		};
	}
}