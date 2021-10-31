#pragma once

namespace SIByL
{
	namespace SRenderGraph
	{
		class RenderPass :public std::enable_shared_from_this<RenderPass>
		{
		public:
			using SharedPtr = Ref<RenderPass>;
			virtual ~RenderPass() = default;
			
		protected:
			friend class RenderGraph;
			RenderPass() = default;

		protected:
			std::string mName;
			std::function<void(void)> mPassChangedCB = [] {};
		};
	}
}