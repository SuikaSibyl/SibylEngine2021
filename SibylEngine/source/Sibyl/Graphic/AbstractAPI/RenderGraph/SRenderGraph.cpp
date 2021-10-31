#include "SIByLpch.h"
#include "SRenderGraph.h"

namespace SIByL
{
	namespace SRenderGraph
	{
        std::vector<RenderGraph*> gRenderGraphs;

        RenderGraph::SharedPtr RenderGraph::create(const std::string& name)
        {
            return SharedPtr(new RenderGraph(name));
        }

        RenderGraph::RenderGraph(const std::string& name)
            : mName(name)
        {
            mpGraph = DirectedGraph::create();
            mpPassDictionary = InternalDictionary::create();
            gRenderGraphs.push_back(this);
            //onResize(gpFramework->getTargetFbo().get());
        }

        RenderGraph::~RenderGraph()
        {
            auto it = std::find(gRenderGraphs.begin(), gRenderGraphs.end(), this);
            assert(it != gRenderGraphs.end());
            gRenderGraphs.erase(it);
        }

        uint32_t RenderGraph::getPassIndex(const std::string& name) const
        {
            auto it = mNameToIndex.find(name);
            return (it == mNameToIndex.end()) ? kInvalidIndex : it->second;
        }

		uint32_t RenderGraph::addPass(const RenderPass::SharedPtr& pPass, const std::string& passName)
		{
            SIByL_CORE_ASSERT(pPass, "RenderGraph::addPass receives NULL RenderPass!");
            uint32_t passIndex = getPassIndex(passName);
            if (passIndex != kInvalidIndex)
            {
                // The pass already exist
                SIByL_CORE_ERROR("Pass named '" + passName + "' already exists. Ignoring call");
                return kInvalidIndex;
            }
            else
            {
                // Add the pass to graph & map
                passIndex = mpGraph->addNode();
                mNameToIndex[passName] = passIndex;
            }

            // If the pass changed, recompile the render graph
            pPass->mPassChangedCB = [this]() { mRecompile = true; };
            pPass->mName = passName;

            //if (mpScene) pPass->setScene(gpDevice->getRenderContext(), mpScene);
            mNodeData[passIndex] = { passName, pPass };
            mRecompile = true;
            return passIndex;
		}

        void RenderGraph::removePass(const std::string& name)
        {
            uint32_t index = getPassIndex(name);
            if (index == kInvalidIndex)
            {
                SIByL_CORE_WARN("Can't remove pass '" + name + "'. Pass doesn't exist");
                return;
            }

            // Unmark graph outputs that belong to this pass
            // Because the way std::vector works, we can't call unmarkOutput() immediately, so we store the outputs in a vector
            std::vector<std::string> outputsToDelete;
            const std::string& outputPrefix = name + '.';
            for (auto& o : mOutputs)
            {
                if (o.nodeId == index) outputsToDelete.push_back(outputPrefix + o.field);
            }

            // Remove all the edges, indices and pass-data associated with this pass
            for (const auto& name : outputsToDelete) unmarkOutput(name);
            mNameToIndex.erase(name);
            mNodeData.erase(index);
            const auto& removedEdges = mpGraph->removeNode(index);
            for (const auto& e : removedEdges) mEdgeData.erase(e);
            mRecompile = true;
        }

        const RenderPass::SharedPtr& RenderGraph::getPass(const std::string& name) const
        {
            uint32_t index = getPassIndex(name);
            if (index == kInvalidIndex)
            {
                static RenderPass::SharedPtr pNull;
                SIByL_CORE_ERROR("RenderGraph::getRenderPass() - can't find a pass named '" + name + "'");
                return pNull;
            }
            return mNodeData.at(index).pPass;
        }

        using str_pair = std::pair<std::string, std::string>;

        template<bool input>
        static bool checkRenderPassIoExist(RenderPass* pPass, const std::string& name)
        {
            //RenderPassReflection reflect = pPass->reflect({});
            //for (size_t i = 0; i < reflect.getFieldCount(); i++)
            //{
            //    const auto& f = *reflect.getField(i);
            //    if (f.getName() == name)
            //    {
            //        return input ? is_set(f.getVisibility(), RenderPassReflection::Field::Visibility::Input) : is_set(f.getVisibility(), RenderPassReflection::Field::Visibility::Output);
            //    }
            //}

            return false;
        }

        static str_pair parseFieldName(const std::string& fullname)
        {
            str_pair strPair;
            if (std::count(fullname.begin(), fullname.end(), '.') == 0)
            {
                // No field name
                strPair.first = fullname;
            }
            else
            {
                size_t dot = fullname.find_last_of('.');
                strPair.first = fullname.substr(0, dot);
                strPair.second = fullname.substr(dot + 1);
            }
            return strPair;
        }

        template<bool input>
        static RenderPass* getRenderPassAndNamePair(const RenderGraph* pGraph, const std::string& fullname, const std::string& errorPrefix, str_pair& nameAndField)
        {
            nameAndField = parseFieldName(fullname);

            RenderPass* pPass = pGraph->getPass(nameAndField.first).get();
            if (!pPass)
            {
                SIByL_CORE_ERROR(errorPrefix + " - can't find render-pass named '" + nameAndField.first + "'");
                return nullptr;
            }

            if (nameAndField.second.size() && checkRenderPassIoExist<input>(pPass, nameAndField.second) == false)
            {
                SIByL_CORE_ERROR(errorPrefix + "- can't find field named '" + nameAndField.second + "' in render-pass '" + nameAndField.first + "'");
                return nullptr;
            }
            return pPass;
        }

        void RenderGraph::unmarkOutput(const std::string& name)
        {
            //str_pair strPair;
            //const auto& pPass = getRenderPassAndNamePair<false>(this, name, "RenderGraph::unmarkGraphOutput()", strPair);
            //if (pPass == nullptr) return;

            //GraphOut removeMe;
            //removeMe.field = strPair.second;
            //removeMe.nodeId = mNameToIndex[strPair.first];

            //for (size_t i = 0; i < mOutputs.size(); i++)
            //{
            //    if (mOutputs[i].nodeId == removeMe.nodeId && mOutputs[i].field == removeMe.field)
            //    {
            //        mOutputs.erase(mOutputs.begin() + i);
            //        mRecompile = true;
            //        return;
            //    }
            //}
        }

        static bool checkMatchingEdgeTypes(const std::string& srcField, const std::string& dstField)
        {
            if (srcField.empty() && dstField.empty()) return true;
            if (dstField.size() && dstField.size()) return true;
            return false;
        }

        uint32_t RenderGraph::addEdge(const std::string& src, const std::string& dst)
        {
            EdgeData newEdge;
            str_pair srcPair, dstPair;
            const auto& pSrc = getRenderPassAndNamePair<false>(this, src, "Invalid src string in RenderGraph::addEdge()", srcPair);
            const auto& pDst = getRenderPassAndNamePair<true>(this, dst, "Invalid dst string in RenderGraph::addEdge()", dstPair);
            newEdge.srcField = srcPair.second;
            newEdge.dstField = dstPair.second;

            if (pSrc == nullptr || pDst == nullptr) return kInvalidIndex;
            if (checkMatchingEdgeTypes(newEdge.srcField, newEdge.dstField) == false)
            {
                SIByL_CORE_ERROR("RenderGraph::addEdge() - can't add the edge [" + src + ", " + dst + "]. \
                    One of the nodes is a resource while the other is a pass. \
                    Can't tell if you want a data-dependency or an execution-dependency");
                return kInvalidIndex;
            }

            uint32_t srcIndex = mNameToIndex[srcPair.first];
            uint32_t dstIndex = mNameToIndex[dstPair.first];


        }
	}
}