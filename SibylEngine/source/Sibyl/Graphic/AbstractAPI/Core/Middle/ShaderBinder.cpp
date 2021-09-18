#include "SIByLpch.h"
#include "ShaderBinder.h"

#include "Sibyl/Graphic/Core/Renderer/Renderer.h"

#include "Sibyl/Graphic/AbstractAPI/Core/Bottom/RootSignature.h"
#include "Platform/DirectX12/AbstractAPI/Middle/DX12ShaderBinder.h"
#include "Platform/OpenGL/AbstractAPI/Middle/OpenGLShaderBinder.h"

namespace SIByL
{
	//////////////////////////////////////////////
	///			Shader Constants Buffer			//
	//////////////////////////////////////////////

	Ref<ShaderConstantsBuffer> ShaderConstantsBuffer::Create(ShaderConstantsDesc* desc)
	{
		switch (Renderer::GetRaster())
		{
		case RasterRenderer::OpenGL: return CreateRef<OpenGLShaderConstantsBuffer>(desc);; break;
		case RasterRenderer::DirectX12: return CreateRef<DX12ShaderConstantsBuffer>(desc); break;
		case RasterRenderer::CpuSoftware: return nullptr; break;
		case RasterRenderer::GpuSoftware: return nullptr; break;
		default: return nullptr; break;
		}
		return nullptr;
	}

	//////////////////////////////////////////////
	///			Shader Resource Buffer			//
	//////////////////////////////////////////////
	
	Ref<ShaderResourcesBuffer> ShaderResourcesBuffer::Create(ShaderResourcesDesc* desc, RootSignature* rs)
	{
		switch (Renderer::GetRaster())
		{
		case RasterRenderer::OpenGL: return CreateRef<OpenGLShaderResourcesBuffer>(desc, rs); break;
		case RasterRenderer::DirectX12: return CreateRef<DX12ShaderResourcesBuffer>(desc, rs); break;
		case RasterRenderer::CpuSoftware: return nullptr; break;
		case RasterRenderer::GpuSoftware: return nullptr; break;
		default: return nullptr; break;
		}
		return nullptr;
	}

	//////////////////////////////////////////////
	///				Shader Binder				//
	//////////////////////////////////////////////
	Ref<ShaderBinder> ShaderBinder::Create(const ShaderBinderDesc& desc)
	{
		switch (Renderer::GetRaster())
		{
		case RasterRenderer::OpenGL: return std::make_shared<OpenGLShaderBinder>(desc);; break;
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
		int paraIndex = 0;
		m_ShaderConstantDescs = new ShaderConstantsDesc[desc.ConstantBufferCount()];
		for (auto constantBuffer : desc.m_ConstantBufferLayouts)
		{
			m_ShaderConstantDescs[cbIndex].Size = constantBuffer.GetStide();
			for (auto bufferElement : constantBuffer)
			{
				m_ConstantsMapper.InsertConstant(bufferElement, cbIndex);
				m_ShaderConstantDescs[cbIndex].Mapper.InsertConstant(bufferElement, cbIndex);
			}
			cbIndex++; paraIndex++;
		}

		int srIndex = 0;
		int innerIndex = 0;
		for (auto resourceBuffer : desc.m_TextureBufferLayouts)
		{
			innerIndex = 0;
			for (auto bufferElement : resourceBuffer)
			{
				m_ShaderResourceDescs.Mapper.InsertResource({ bufferElement.Name,bufferElement.Type,paraIndex,innerIndex, "PROCEDURE=White"});
				m_ResourcesMapper.InsertResource({ bufferElement.Name,bufferElement.Type,paraIndex,innerIndex });
				innerIndex++;
			}
			srIndex++; paraIndex++;
		}
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

	void ResourcesMapper::InsertResource(const ShaderResourceItem& element)
	{
		// If the name already exists
		std::string name = element.Name;
		if (m_Mapper.find(name) != m_Mapper.end())
			SIByL_CORE_ERROR("Duplicated Shader Constant Name!");
		// else insert the item
		m_Mapper.insert({ name, element });
	}

	bool ResourcesMapper::FetchResource(std::string name, ShaderResourceItem& buffer)
	{
		// If the name already exists
		if (m_Mapper.find(name) == m_Mapper.end())
		{
			SIByL_CORE_ERROR("Shader Resource \'{0}\'Not Found!", name);
			return false;
		}
		else
		{
			buffer = m_Mapper[name];
			return true;
		}
	}

	bool ResourcesMapper::SetTextureID(std::string name, std::string ID)
	{
		// If the name already exists
		if (m_Mapper.find(name) == m_Mapper.end())
		{
			SIByL_CORE_ERROR("Shader Resource \'{0}\'Not Found!", name);
			return false;
		}
		else
		{
			m_Mapper[name].TextureID = ID;
			return true;
		}
	}

}