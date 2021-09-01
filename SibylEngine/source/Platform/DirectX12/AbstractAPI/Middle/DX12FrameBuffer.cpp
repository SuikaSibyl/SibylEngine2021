#include "SIByLpch.h"
#include "DX12FrameBuffer.h"

#include "Platform/DirectX12/AbstractAPI/Middle/DX12Texture.h"

namespace SIByL
{
	///////////////////////////////////////////////////////////////////////////
	///                      Constructors / Destructors                     ///
	///////////////////////////////////////////////////////////////////////////
	
	// Create From Descriptor
	// ----------------------------------------------------------
	DX12FrameBuffer::DX12FrameBuffer(const FrameBufferDesc& desc)
		:m_Desc(desc)
	{

	}

	DX12FrameBuffer::~DX12FrameBuffer()
	{

	}

	///////////////////////////////////////////////////////////////////////////
	///						        Manipulator		                        ///
	///////////////////////////////////////////////////////////////////////////
	void DX12FrameBuffer::Bind()
	{

	}

	void DX12FrameBuffer::Unbind()
	{

	}

	void DX12FrameBuffer::Resize(uint32_t width, uint32_t height)
	{
		
	}

	void DX12FrameBuffer::ClearBuffer()
	{

	}

	void DX12FrameBuffer::ClearRgba()
	{

	}

	void DX12FrameBuffer::ClearDepthStencil()
	{

	}

	///////////////////////////////////////////////////////////////////////////
	///                                Caster                               ///
	///////////////////////////////////////////////////////////////////////////

	Ref<Texture2D> DX12FrameBuffer::ColorAsTexutre()
	{
		Ref<Texture2D> texture = nullptr;
		//texture.reset(new OpenGLTexture2D(m_TextureObject, m_Desc.Width,
		//	m_Desc.Height, m_Desc.Channel, Texture2D::Format::R24G8));
		return texture;
	}

	Ref<Texture2D> DX12FrameBuffer::DepthStencilAsTexutre()
	{
		Ref<Texture2D> texture = nullptr;
		//texture.reset(new OpenGLTexture2D(m_TextureObject, m_Desc.Width,
		//	m_Desc.Height, m_Desc.Channel, Texture2D::Format::R24G8));
		return texture;
	}
}