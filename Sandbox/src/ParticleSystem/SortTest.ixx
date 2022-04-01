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
		GFX::RDG::NodeHandle globalHistogram;

		GFX::RDG::NodeHandle sortHistogramNaive1_32;
		GFX::RDG::NodeHandle sortHistogramIntegrate1_32;
		GFX::RDG::NodeHandle sortInit;
		GFX::RDG::NodeHandle sortPass;

		MemScope<RHI::IShader> shaderHistogramNaive1_32;
		MemScope<RHI::IShader> shaderHistogramIntegrate1_32;
		MemScope<RHI::IShader> shaderSortInit;
		MemScope<RHI::IShader> shaderSortPass;

		RHI::IResourceFactory* factory;
		uint32_t elementCount;
		uint32_t threadPerWorkgroup = 1024;
		uint32_t elementPerWorkgroup = 1024;
		uint32_t tileSize = 2048;
		uint32_t tilePerBlock = 1;
		uint32_t elementPerBlock = tileSize * tilePerBlock;
		uint32_t elementReducedSize;
		uint32_t bitsPerDigit = 1;
		uint32_t possibleDigitValue = 1 << bitsPerDigit;
		uint32_t passNum = 32 / bitsPerDigit;

		uint32_t activeBlockNum = ((elementCount + elementPerBlock - 1) / elementPerBlock);
	};

	SortTest::SortTest(RHI::IResourceFactory* factory, uint32_t element_count)
		:factory(factory), elementCount(element_count)
	{
		elementReducedSize = ((elementCount + elementPerWorkgroup - 1) / elementPerWorkgroup);

		// load precomputed samples for particle position initialization
		//shaderHistogramNaive1_32 = factory->createShaderFromBinaryFile("cluster/radix_sort_histogram_naive_1_32.spv", { RHI::ShaderStage::COMPUTE,"main" });
		shaderHistogramNaive1_32 = factory->createShaderFromBinaryFile("cluster/radix_sort_histogram_subgroup_1_32.spv", { RHI::ShaderStage::COMPUTE,"main" });
		shaderHistogramIntegrate1_32 = factory->createShaderFromBinaryFile("cluster/radix_sort_histogram_integrate_1_32.spv", { RHI::ShaderStage::COMPUTE,"main" });
		shaderSortInit = factory->createShaderFromBinaryFile("cluster/radix_sort_test_initializer.spv", { RHI::ShaderStage::COMPUTE,"main" });
		shaderSortPass = factory->createShaderFromBinaryFile("cluster/radix_sort_onesweep.spv", { RHI::ShaderStage::COMPUTE,"main" });
	}

	auto SortTest::registerResources(GFX::RDG::RenderGraphBuilder* builder) noexcept -> void
	{
		// Resources
		inputKeys = builder->addStorageBuffer(sizeof(uint32_t) * elementCount, "Input Keys");
		sortedIndexWithDoubleBuffer = builder->addStorageBuffer(sizeof(uint32_t) * elementCount * 2, "Sorted Indices Double-Buffer");
		offsetFromDigitStartsAggregate = builder->addStorageBuffer(sizeof(uint32_t) * GRIDSIZE(elementCount, (tileSize * tilePerBlock)), "Offset (Aggregate)");
		offsetFromDigitStartsAggregate = builder->addStorageBuffer(sizeof(uint32_t) * GRIDSIZE(elementCount, (tileSize * tilePerBlock)), "Offset (Prefix)");
		intermediateHistogram = builder->addStorageBuffer(sizeof(uint32_t) * passNum * possibleDigitValue * elementReducedSize, "Block-Wise Sum");
		globalHistogram = builder->addStorageBuffer(sizeof(uint32_t) * passNum * possibleDigitValue, "Global Histogram");
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
		sortInit = builder->addComputePassBackPool(shaderSortInit.get(), { inputKeys, sortedIndexWithDoubleBuffer }, "Sort Init", sizeof(unsigned int));

		// Create Sort Pass
		sortHistogramNaive1_32 = builder->addComputePassBackPool(shaderHistogramNaive1_32.get(), { inputKeys, intermediateHistogram }, "Naive Histogram Pass 1", sizeof(unsigned int));
		sortHistogramIntegrate1_32 = builder->addComputePassBackPool(shaderHistogramIntegrate1_32.get(), { intermediateHistogram, globalHistogram }, "Histogram Pass 2", sizeof(unsigned int));
		sortPass = builder->addComputePassBackPool(shaderSortPass.get(), { inputKeys, sortedIndexWithDoubleBuffer, offsetFromDigitStartsAggregate, offsetFromDigitStartsAggregate, globalHistogram }, "Sort Pass", sizeof(unsigned int));

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