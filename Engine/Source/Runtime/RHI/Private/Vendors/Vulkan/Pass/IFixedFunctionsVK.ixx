module;
#include <vulkan/vulkan.h>
#include <vector>
export module RHI.IFixedFunctions.VK;
import RHI.IFixedFunctions;

namespace SIByL
{
	namespace RHI
	{
		export class IVertexLayoutVK :public IVertexLayout
		{
		public:
			IVertexLayoutVK() = default;
			IVertexLayoutVK(IVertexLayoutVK&&) = default;
			virtual ~IVertexLayoutVK() = default;

			auto getVkInputState() noexcept -> VkPipelineVertexInputStateCreateInfo*;
		private:
			auto createVkInputState() noexcept -> void;
			VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		};

		export class IInputAssemblyVK :public IInputAssembly
		{
		public:
			IInputAssemblyVK() = default;
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
			IViewportsScissorsVK() = default;
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
			IRasterizerVK() = default;
			IRasterizerVK(IRasterizerVK&&) = default;
			virtual ~IRasterizerVK() = default;

			auto getVkPipelineRasterizationStateCreateInfo() noexcept
				-> VkPipelineRasterizationStateCreateInfo*;
		private:
			auto createRasterizerStateInfo() noexcept -> void;
			VkPipelineRasterizationStateCreateInfo rasterizer{};
		};

		export class IMultisamplingVK :public IMultisampling
		{
		public:
			IMultisamplingVK() = default;
			IMultisamplingVK(IMultisamplingVK&&) = default;
			virtual ~IMultisamplingVK() = default;

			auto getVkPipelineMultisampleStateCreateInfo() noexcept
				-> VkPipelineMultisampleStateCreateInfo*;
		private:
			auto createMultisampingInfo() noexcept -> void;
			VkPipelineMultisampleStateCreateInfo multisampling{};
		};

		export class IDepthStencilVK :public IDepthStencil
		{
		public:
			IDepthStencilVK() = default;
			IDepthStencilVK(IDepthStencilVK&&) = default;
			virtual ~IDepthStencilVK() = default;

			auto getVkPipelineDepthStencilStateCreateInfo() noexcept
				-> VkPipelineDepthStencilStateCreateInfo*;

		private:
			VkPipelineDepthStencilStateCreateInfo depthStencil{};
		};

		export class IColorBlendingVK :public IColorBlending
		{
		public:
			IColorBlendingVK() = default;
			IColorBlendingVK(IColorBlendingVK&&) = default;
			virtual ~IColorBlendingVK() = default;

			auto getVkPipelineColorBlendAttachmentState() noexcept
				-> VkPipelineColorBlendAttachmentState*;
			auto getVkPipelineColorBlendStateCreateInfo() noexcept
				-> VkPipelineColorBlendStateCreateInfo*;
		private:
			auto createColorBlendObjects() noexcept -> void;
			VkPipelineColorBlendAttachmentState colorBlendAttachment{};
			VkPipelineColorBlendStateCreateInfo colorBlending{};
		};

		export class IDynamicStateVK :public IDynamicState
		{
		public:
			IDynamicStateVK() = default;
			IDynamicStateVK(IDynamicStateVK&&) = default;
			virtual ~IDynamicStateVK() = default;

		private:
			auto createDynamicState() noexcept -> void;
			std::vector<VkDynamicState> dynamicStates;
			VkPipelineDynamicStateCreateInfo dynamicState{};
		};

	}
}
