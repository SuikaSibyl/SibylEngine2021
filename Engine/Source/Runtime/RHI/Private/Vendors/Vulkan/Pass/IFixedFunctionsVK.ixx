module;
#include <vulkan/vulkan.h>
#include <vector>
export module RHI.IFixedFunctions.VK;
import RHI.IFixedFunctions;
import RHI.IEnum;
import RHI.IBuffer;

namespace SIByL
{
	namespace RHI
	{
		export class IVertexLayoutVK :public IVertexLayout
		{
		public:
			IVertexLayoutVK(BufferLayout& layout);
			IVertexLayoutVK(IVertexLayoutVK&&) = default;
			virtual ~IVertexLayoutVK() = default;

			auto getVkInputState() noexcept -> VkPipelineVertexInputStateCreateInfo*;
		private:
			auto createVkInputState(BufferLayout& layout) noexcept -> void;
			VkVertexInputBindingDescription bindingDescription{};
			std::vector<VkVertexInputAttributeDescription> attributeDescriptions{};
			VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		};

		export class IInputAssemblyVK :public IInputAssembly
		{
		public:
			IInputAssemblyVK(TopologyKind topology_kind);
			IInputAssemblyVK(IInputAssemblyVK&&) = default;
			virtual ~IInputAssemblyVK() = default;

			auto createVkInputAssembly() noexcept -> void;
			auto getVkInputAssembly() noexcept -> VkPipelineInputAssemblyStateCreateInfo*;
		private:
			VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		};

		export class IViewportsScissorsVK :public IViewportsScissors
		{
		public:
			IViewportsScissorsVK(
				unsigned int width_viewport, 
				unsigned int height_viewport,
				unsigned int width_scissor,
				unsigned int height_scissor);
			IViewportsScissorsVK(IViewportsScissorsVK&&) = default;
			virtual ~IViewportsScissorsVK() = default;

			auto getVkPipelineViewportStateCreateInfo() noexcept -> VkPipelineViewportStateCreateInfo*;
		private:
			auto createVkPipelineViewportStateCreateInfo
			(float width, float height, VkExtent2D* extend) noexcept -> void;

			VkViewport viewport{};
			VkRect2D scissor{};
			VkPipelineViewportStateCreateInfo viewportState{};
		};

		export class IRasterizerVK :public IRasterizer
		{
		public:
			IRasterizerVK(RasterizerDesc const& desc);
			IRasterizerVK(IRasterizerVK&&) = default;
			virtual ~IRasterizerVK() = default;

			auto getVkPipelineRasterizationStateCreateInfo() noexcept
				-> VkPipelineRasterizationStateCreateInfo*;
		private:
			auto createRasterizerStateInfo(RasterizerDesc const& desc) noexcept -> void;
			VkPipelineRasterizationStateCreateInfo rasterizer{};
		};

		export class IMultisamplingVK :public IMultisampling
		{
		public:
			IMultisamplingVK(MultiSampleDesc const& desc);
			IMultisamplingVK(IMultisamplingVK&&) = default;
			virtual ~IMultisamplingVK() = default;

			auto getVkPipelineMultisampleStateCreateInfo() noexcept
				-> VkPipelineMultisampleStateCreateInfo*;
		private:
			auto createMultisampingInfo(MultiSampleDesc const& desc) noexcept -> void;
			VkPipelineMultisampleStateCreateInfo multisampling{};
		};

		export class IDepthStencilVK :public IDepthStencil
		{
		public:
			IDepthStencilVK(DepthStencilDesc const& desc);
			IDepthStencilVK(IDepthStencilVK&&) = default;
			virtual ~IDepthStencilVK() = default;

			auto getVkPipelineDepthStencilStateCreateInfo() noexcept
				-> VkPipelineDepthStencilStateCreateInfo*;

		private:
			bool initialized = false;
			VkPipelineDepthStencilStateCreateInfo depthStencil{};
		};

		export class IColorBlendingVK :public IColorBlending
		{
		public:
			IColorBlendingVK(ColorBlendingDesc const& desc);
			IColorBlendingVK(IColorBlendingVK&&) = default;
			virtual ~IColorBlendingVK() = default;

			auto getVkPipelineColorBlendAttachmentState() noexcept
				-> VkPipelineColorBlendAttachmentState*;
			auto getVkPipelineColorBlendStateCreateInfo() noexcept
				-> VkPipelineColorBlendStateCreateInfo*;
		private:
			auto createColorBlendObjects(ColorBlendingDesc const& desc) noexcept -> void;
			VkPipelineColorBlendAttachmentState colorBlendAttachment{};
			VkPipelineColorBlendStateCreateInfo colorBlending{};
		};

		export class IDynamicStateVK :public IDynamicState
		{
		public:
			IDynamicStateVK(std::vector<PipelineState> const& states);
			IDynamicStateVK(IDynamicStateVK&&) = default;
			virtual ~IDynamicStateVK() = default;

		private:
			auto createDynamicState(std::vector<PipelineState> const& states) noexcept -> void;
			std::vector<VkDynamicState> dynamicStates;
			VkPipelineDynamicStateCreateInfo dynamicState{};
		};

	}
}
