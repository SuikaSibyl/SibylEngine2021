#include "SIByLpch.h"
#include "ShaderBinder.h"

#include "Sibyl/Renderer/Renderer.h"

#include "Platform/DirectX12/Renderer/DX12ShaderBinder.h"

namespace SIByL
{
	Ref<ShaderBinder> ShaderBinder::Create(const ShaderBinderDesc& desc)
	{
		switch (Renderer::GetRaster())
		{
		case RasterRenderer::OpenGL: return nullptr; break;
		case RasterRenderer::DirectX12: return std::make_shared<DX12ShaderBinder>(desc); break;
		case RasterRenderer::CpuSoftware: return nullptr; break;
		case RasterRenderer::GpuSoftware: return nullptr; break;
		default: return nullptr; break;
		}
		return nullptr;
	}

	void ShaderBinder::InitMappers(const ShaderBinderDesc& desc)
	{
		int cbIndex = 0;
		for (auto constantBuffer : desc.m_ConstantBufferLayouts)
		{
			for (auto bufferElement : constantBuffer)
			{
				m_ConstantsMapper.InsertConstant(bufferElement, cbIndex);
			}
			cbIndex++;
		}

		//for (auto resourceBuffer : desc.m_TextureBufferLayouts)
		//{
		//	for (auto bufferElement : resourceBuffer)
		//	{
		//		//bufferElement.Name;
		//	}
		//}
	}

	void ConstantsMapper::InsertConstant(const BufferElement& element, int CBIndex)
	{
		// If the name already exists
		std::string name = element.Name;
		if (m_Mapper.find(name) != m_Mapper.end())
			SIByL_CORE_ERROR("Duplicated Shader Constant Name!");
		// else insert the item
		m_Mapper.insert({ name, ShaderConstantItem{name, element.Type, CBIndex, element.Offset} });
	}

	bool ConstantsMapper::FetchConstant(std::string name, ShaderConstantItem& buffer)
	{
		// If the name already exists
		if (m_Mapper.find(name) == m_Mapper.end())
		{
			SIByL_CORE_ERROR("Shader Constant \'{0}\'Not Found!", name);
			return false;
		}
		else
		{
			buffer = m_Mapper[name];
			return true;
		}
	}
}