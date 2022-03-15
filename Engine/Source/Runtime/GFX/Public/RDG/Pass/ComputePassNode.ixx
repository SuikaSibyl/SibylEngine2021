module;
#include <vector>
export module GFX.RDG.ComputePassNode;
import RHI.IShader;
import RHI.IPipeline;
import RHI.IFactory;
import RHI.IDescriptorSet;
import RHI.IDescriptorSetLayout;
import Core.MemoryManager;
import GFX.RDG.Common;

namespace SIByL::GFX::RDG
{
	export struct ComputePassInfo
	{
		RHI::DescriptorSetLayoutDesc dsLayoutDesc;

	};

	export struct ComputePassNode :public PassNode
	{
	public:
		ComputePassNode(void* graph, RHI::IShader* shader, std::vector<NodeHandle>&& ios, uint32_t const& constant_size = 0);
		virtual auto onBuild(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;
		auto execute(RHI::ICommandBuffer* buffer, unsigned int x, unsigned int y, unsigned int z, unsigned int frame) noexcept -> void;

		template<class T>
		auto executeWithConstant(RHI::ICommandBuffer* buffer, unsigned int x, unsigned int y, unsigned int z, unsigned int frame, T const& constant) noexcept -> void
		{
			RHI::IDescriptorSet* compute_tmp_set = compute_descriptorSets[frame].get();
			buffer->cmdBindComputePipeline(pipeline.get());
			buffer->cmdBindDescriptorSets(RHI::PipelineBintPoint::COMPUTE,
				compute_pipeline_layout.get(), 0, 1, &compute_tmp_set, 0, nullptr);
			buffer->cmdPushConstants(compute_pipeline_layout.get(), RHI::ShaderStage::COMPUTE, sizeof(T), const_cast<T*>(&constant));
			buffer->cmdDispatch(x, y, z);
		}

	private:
		uint32_t constantSize;
		RHI::IShader* shader;
		std::vector<NodeHandle> ios;
		MemScope<RHI::IPipeline> pipeline;
		MemScope<RHI::IPipelineLayout> compute_pipeline_layout;
		std::vector<MemScope<RHI::IDescriptorSet>> compute_descriptorSets;
	};
}