#pragma once

namespace SIByL
{
	///////////////////////////////////////////
	// 	      Basic Graph Data Structure	 //
	///////////////////////////////////////////
	class SRGNode
	{

	};

	class SRGEdge
	{

	};

	class SRGRegistry;
	class SRGPassExecutor
	{
	public:
		virtual void Execute(SRGRegistry& registry, void* context) = 0;
	};

	class SRGPassNode :public SRGNode
	{
		SRGPassExecutor* m_PassExecutor;
	};

	class Edge;
	class SRenderDependencyGraph
	{
		std::vector<SRGNode*> m_Nodes;
		std::vector<Edge*> m_Edges;
	};

	class SRGVirtualResource;
	class SRGResourceNode :public SRGNode
	{
		uint64_t m_Index;
		uint64_t m_ParentIndex;

		std::vector<SRGPassNode*> m_Readers;
		SRGPassNode* m_Writer;

		SRGVirtualResource* m_Resource;
	};
}