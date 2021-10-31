#pragma once

#include "SRenderPass.h"
#include "Sibyl/Basic/Algorithms/DirectedGraph.h"
#include "Sibyl/Basic/Utils/InternalDirectory.h"

namespace SIByL
{
	namespace SRenderGraph
	{
		class RenderPass;

		class RenderGraph
		{
		public:
			using SharedPtr = std::shared_ptr<RenderGraph>;
			//static const FileDialogFilterVec kFileExtensionFilters;

			static const uint32_t kInvalidIndex = -1;
			
			~RenderGraph();

			/** Create a new render graph.
				\param[in] name Name of the render graph.
				\return New object, or throws an exception if creation failed.
			*/
			static SharedPtr create(const std::string& name = "");



			/** Add a render-pass. The name has to be unique, otherwise the call will be ignored
			*/
			uint32_t addPass(const RenderPass::SharedPtr& pPass, const std::string& passName);

			/** Get a render-pass
			*/
			const RenderPass::SharedPtr& getPass(const std::string& name) const;

			/** Remove a render-pass. You need to make sure the edges are still valid after the node was removed
			*/
			void removePass(const std::string& name);

			/** Insert an edge from a render-pass' output into a different render-pass input.
				The render passes must be different, the graph must be a DAG.
				There are 2 types of edges:
				- Data dependency edge - Connecting as pass` output resource to another pass` input resource.
										 The src/dst strings have the format `renderPassName.resourceName`, where the `renderPassName` is the name used in `addPass()` and the `resourceName` is the resource-name as described by the render-pass object
				- Execution dependency edge - As the name implies, it creates an execution dependency between 2 passes, even if there's no data dependency. You can use it to control the execution order of the graph, or to force execution of passes which have no inputs/outputs.
											  The src/dst string are `srcPass` and `dstPass` as used in `addPass()`

				Note that data-dependency edges may be optimized out of the execution, if they are determined not to influence the requested graph-output. Execution-dependency edges are never optimized and will always execute
			*/
			uint32_t addEdge(const std::string& src, const std::string& dst);


			/** Return the index of a pass from a name, or kInvalidIndex if the pass doesn't exists
			*/
			uint32_t RenderGraph::getPassIndex(const std::string& name) const;

			/** Unmark a graph output
				The name has the format `renderPassName.resourceName`. You can also use `renderPassName` which will allocate all the render-pass outputs
			*/
			void unmarkOutput(const std::string& name);

		protected:


		private:

			RenderGraph(const std::string& name);
			std::string mName;

			struct EdgeData
			{
				std::string srcField;
				std::string dstField;
			};

			struct NodeData
			{
				std::string name;
				RenderPass::SharedPtr pPass;
			};

			std::unordered_map<std::string, uint32_t> mNameToIndex;
			DirectedGraph::SharedPtr mpGraph;
			std::unordered_map<uint32_t, EdgeData> mEdgeData;
			std::unordered_map<uint32_t, NodeData> mNodeData;

			struct GraphOut
			{
				uint32_t nodeId;
				std::string field;

				bool operator==(const GraphOut& other) const
				{
					if (nodeId != other.nodeId) return false;
					if (field != other.field) return false;
					return true;
				}

				bool operator!=(const GraphOut& other) const { return !(*this == other); }
			};

			std::vector<GraphOut> mOutputs;
			bool isGraphOutput(const GraphOut& graphOut) const;

			InternalDictionary::SharedPtr mpPassDictionary;
			bool mRecompile = false;
			
		};
	}
}