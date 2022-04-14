module;
#include <cstdint>
#include <filesystem>
#include <vector>
#include <functional>
#define GRIDSIZE(x,ThreadSize) ((x+ThreadSize - 1)/ThreadSize)
export module Demo.SortTest;
import Core.Buffer;
import Core.Cache;
import Core.Image;
import Core.File;
import Core.Time;
import Core.MemoryManager;
import RHI.IEnum;
import RHI.IFactory;
import RHI.IBuffer;
import RHI.ICommandBuffer;
import RHI.IShader;
import RHI.ISampler;
import RHI.IStorageBuffer;
import RHI.ITexture;
import RHI.ITextureView;
import GFX.RDG.Common;
import GFX.RDG.RenderGraph;
import GFX.RDG.StorageBufferNode;
import GFX.RDG.ComputePassNode;

import ParticleSystem.ParticleSystem;

namespace SIByL::Demo
{
	export class SortTest
	{
	public:
		SortTest() = default;
		SortTest(RHI::IResourceFactory * factory, uint32_t element_count);

		auto registerResources(GFX::RDG::RenderGraphBuilder* builder) noexcept -> void;
		auto registerUpdatePasses(GFX::RDG::RenderGraphBuilder* builder) noexcept -> void;

		// Resource Nodes Handles --------------------------
		GFX::RDG::NodeHandle inputKeys;
		GFX::RDG::NodeHandle sortedIndexWithDoubleBuffer;
		GFX::RDG::NodeHandle offsetFromDigitStartsAggregate;
		GFX::RDG::NodeHandle offsetFromDigitStartPrefix;
		GFX::RDG::NodeHandle intermediateHistogram;
		GFX::RDG::NodeHandle intermediateHistogramLookBack;
		GFX::RDG::NodeHandle globalHistogram;
		GFX::RDG::NodeHandle globalCounter;
		// Only for Debug
		GFX::RDG::NodeHandle sortedKeys;
		GFX::RDG::NodeHandle debugInfo;

		GFX::RDG::NodeHandle sortHistogramSubgroup_8_4;
		GFX::RDG::NodeHandle sortHistogramIntegrate_8_4;
		GFX::RDG::NodeHandle sortInit;
		GFX::RDG::NodeHandle sortPass;
		GFX::RDG::NodeHandle sortPass_0;
		GFX::RDG::NodeHandle sortPass_1;
		GFX::RDG::NodeHandle sortPass_2;
		GFX::RDG::NodeHandle sortPass_3;
		GFX::RDG::NodeHandle sortPassClear;
		GFX::RDG::NodeHandle sortShowKeys;

		MemScope<RHI::IShader> shaderHistogramNaive_8_4;
		MemScope<RHI::IShader> shaderHistogramIntegrate_8_4;
		MemScope<RHI::IShader> shaderSortInit;
		MemScope<RHI::IShader> shaderSortPass;
		MemScope<RHI::IShader> shaderSortPassClear;
		// Only for Debug
		MemScope<RHI::IShader> shaderSortShowKeys;

		RHI::IResourceFactory* factory;
		uint32_t elementCount;
		uint32_t threadPerWorkgroup = 256;
		uint32_t elementPerThread = 8;
		uint32_t elementPerWorkgroup = threadPerWorkgroup * elementPerThread;
		uint32_t bitsPerDigit = 8;
		uint32_t tileSize = elementPerWorkgroup;
		uint32_t tilePerBlock = 1;
		uint32_t elementPerBlock = tileSize * tilePerBlock;
		uint32_t elementReducedSize;
		uint32_t possibleDigitValue = 1 << bitsPerDigit;
		uint32_t passNum = 32 / bitsPerDigit;
		uint32_t subgroupHistogramElementPerBlock = 8 * 256;
		

		uint32_t activeBlockNum = ((elementCount + elementPerBlock - 1) / elementPerBlock);
	};

	SortTest::SortTest(RHI::IResourceFactory* factory, uint32_t element_count)
		:factory(factory), elementCount(element_count)
	{
		elementReducedSize = ((elementCount + elementPerWorkgroup - 1) / elementPerWorkgroup);

		// load precomputed samples for particle position initialization
		//shaderHistogramNaive1_32 = factory->createShaderFromBinaryFile("cluster/radix_sort_histogram_naive_1_32.spv", { RHI::ShaderStage::COMPUTE,"main" });
		shaderHistogramNaive_8_4 = factory->createShaderFromBinaryFile("cluster/radix_sort_histogram_subgroup_8_4.spv", { RHI::ShaderStage::COMPUTE,"main" });
		shaderHistogramIntegrate_8_4 = factory->createShaderFromBinaryFile("cluster/radix_sort_histogram_integrate_8_4.spv", { RHI::ShaderStage::COMPUTE,"main" });
		shaderSortInit = factory->createShaderFromBinaryFile("cluster/radix_sort_test_initializer.spv", { RHI::ShaderStage::COMPUTE,"main" });
		shaderSortPass = factory->createShaderFromBinaryFile("cluster/radix_sort_onesweep_template.spv", { RHI::ShaderStage::COMPUTE,"main" });
		shaderSortPassClear = factory->createShaderFromBinaryFile("cluster/radix_sort_onesweep_clear.spv", { RHI::ShaderStage::COMPUTE,"main" });
		shaderSortShowKeys = factory->createShaderFromBinaryFile("cluster/radix_sort_show_keys.spv", { RHI::ShaderStage::COMPUTE,"main" });
	}

	auto SortTest::registerResources(GFX::RDG::RenderGraphBuilder* builder) noexcept -> void
	{
		// Resources
		inputKeys = builder->addStorageBuffer(sizeof(uint32_t) * elementCount, "Input Keys");
		sortedIndexWithDoubleBuffer = builder->addStorageBuffer(sizeof(uint32_t) * elementCount * 2, "Sorted Indices Double-Buffer");
		offsetFromDigitStartsAggregate = builder->addStorageBuffer(sizeof(uint32_t) * possibleDigitValue *  GRIDSIZE(elementCount, (tileSize * tilePerBlock)), "Offset (Aggregate)");
		offsetFromDigitStartPrefix = builder->addStorageBuffer(sizeof(uint32_t) * possibleDigitValue * GRIDSIZE(elementCount, (tileSize * tilePerBlock)), "Offset (Prefix)");
		intermediateHistogram = builder->addStorageBuffer(sizeof(uint32_t) * passNum * possibleDigitValue, "Block-Wise Sum");
		intermediateHistogramLookBack = builder->addStorageBuffer(sizeof(uint32_t) * passNum * possibleDigitValue * GRIDSIZE(elementCount, subgroupHistogramElementPerBlock), "Block-Wise Sum Look Back");
		globalHistogram = builder->addStorageBuffer(sizeof(uint32_t) * passNum * possibleDigitValue, "Global Histogram");
		globalCounter = builder->addStorageBuffer(sizeof(uint32_t) * 2, "Global Counter");

		sortedKeys = builder->addStorageBuffer(sizeof(uint32_t) * elementCount, "Sorted Keys");
		debugInfo = builder->addStorageBuffer(sizeof(uint32_t) * elementCount * 2, "Debug Info");
	}

	struct EmitConstant
	{
		unsigned int emitCount;
		float time;
		float x;
		float y;
	};

	auto SortTest::registerUpdatePasses(GFX::RDG::RenderGraphBuilder* builder) noexcept -> void
	{
		// Create Init Pass
		sortInit = builder->addComputePassBackPool(shaderSortInit.get(), { inputKeys, sortedIndexWithDoubleBuffer, globalCounter }, "Sort Init", sizeof(unsigned int));

		// Create Sort Pass
		sortHistogramSubgroup_8_4 = builder->addComputePassBackPool(shaderHistogramNaive_8_4.get(), { inputKeys, intermediateHistogram, intermediateHistogramLookBack }, "Naive Histogram Pass 1", 0);
		sortHistogramIntegrate_8_4 = builder->addComputePassBackPool(shaderHistogramIntegrate_8_4.get(), { intermediateHistogram, globalHistogram }, "Histogram Pass 2", sizeof(unsigned int));
		sortPass = builder->addComputePassBackPool(shaderSortPass.get(), { inputKeys, sortedIndexWithDoubleBuffer, offsetFromDigitStartsAggregate, offsetFromDigitStartPrefix, globalHistogram, globalCounter, debugInfo }, "Sort Pass", sizeof(unsigned int));
		
		sortPass_0 = builder->addComputePass(shaderSortPass.get(), { inputKeys, sortedIndexWithDoubleBuffer, offsetFromDigitStartsAggregate, offsetFromDigitStartPrefix, globalHistogram, globalCounter, debugInfo }, "Sort Pass", sizeof(unsigned int));
		builder->attached.getComputePassNode(sortPass_0)->customDispatch = [elementCount = elementCount, elementPerBlock = elementPerBlock] (GFX::RDG::ComputePassNode * compute_pass, RHI::ICommandBuffer * commandbuffer, uint32_t flight_idx)
		{ compute_pass->executeWithConstant(commandbuffer, GRIDSIZE(elementCount, elementPerBlock), 1, 1, flight_idx, 0); };

		sortPass_1 = builder->addComputePass(shaderSortPass.get(), { inputKeys, sortedIndexWithDoubleBuffer, offsetFromDigitStartsAggregate, offsetFromDigitStartPrefix, globalHistogram, globalCounter, debugInfo }, "Sort Pass", sizeof(unsigned int));
		builder->attached.getComputePassNode(sortPass_1)->customDispatch = [elementCount = elementCount, elementPerBlock = elementPerBlock](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
		{ compute_pass->executeWithConstant(commandbuffer, GRIDSIZE(elementCount, elementPerBlock), 1, 1, flight_idx, 1); };

		sortPass_2 = builder->addComputePass(shaderSortPass.get(), { inputKeys, sortedIndexWithDoubleBuffer, offsetFromDigitStartsAggregate, offsetFromDigitStartPrefix, globalHistogram, globalCounter, debugInfo }, "Sort Pass", sizeof(unsigned int));
		builder->attached.getComputePassNode(sortPass_2)->customDispatch = [elementCount = elementCount, elementPerBlock = elementPerBlock](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
		{ compute_pass->executeWithConstant(commandbuffer, GRIDSIZE(elementCount, elementPerBlock), 1, 1, flight_idx, 2); };

		sortPass_3 = builder->addComputePass(shaderSortPass.get(), { inputKeys, sortedIndexWithDoubleBuffer, offsetFromDigitStartsAggregate, offsetFromDigitStartPrefix, globalHistogram, globalCounter, debugInfo }, "Sort Pass", sizeof(unsigned int));
		builder->attached.getComputePassNode(sortPass_3)->customDispatch = [elementCount = elementCount, elementPerBlock = elementPerBlock](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
		{ compute_pass->executeWithConstant(commandbuffer, GRIDSIZE(elementCount, elementPerBlock), 1, 1, flight_idx, 3); };

		
		
		sortPassClear = builder->addComputePassBackPool(shaderSortPassClear.get(), { offsetFromDigitStartsAggregate, offsetFromDigitStartPrefix }, "Sort Pass", 0);
		sortShowKeys = builder->addComputePassBackPool(shaderSortShowKeys.get(), { inputKeys, sortedIndexWithDoubleBuffer, sortedKeys }, "Sort Test Pass", 0);

		//emitPass = builder->addComputePass(emitShader, { particleBuffer, counterBuffer, liveIndexBuffer, deadIndexBuffer, emitterVolumeHandle, samplerHandle }, "Particles Emit", sizeof(unsigned int) * 4);
		//builder->attached.getComputePassNode(emitPass)->customDispatch = [&timer = timer](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
		//{
		//	EmitConstant constant_1{ 400000u / 50, (float)timer->getTotalTime(), 0, 1.07 };
		//	compute_pass->executeWithConstant(commandbuffer, 200, 1, 1, flight_idx, constant_1);
		//};
		//GFX::RDG::ComputePassNode* emitPassNode = builder->attached.getComputePassNode(emitPass);
		//emitPassNode->textures = { bakedCurveHandle };

		//// Create Update Pass
		//updatePass = builder->addComputePass(updateShader, { particleBuffer, counterBuffer, liveIndexBuffer, deadIndexBuffer, indirectDrawBuffer }, "Particles Update");
		//builder->attached.getComputePassNode(updatePass)->customDispatch = [](GFX::RDG::ComputePassNode* compute_pass, RHI::ICommandBuffer* commandbuffer, uint32_t flight_idx)
		//{
		//	compute_pass->execute(commandbuffer, 200, 1, 1, flight_idx);
		//};
	}
}