#include "SIByLpch.h"
#include "DX12Texture.h"

#include "Sibyl/Graphic/Core/Texture/Image.h"

#include "Platform/DirectX12/Common/DX12Utility.h"
#include "Platform/DirectX12/Common/DX12Context.h"


namespace SIByL
{
	DX12Texture2D::DX12Texture2D(const std::string& path)
	{
		PROFILE_SCOPE_FUNCTION();
		m_Path = path;
		Image image(path);
		InitFromImage(&image);
	}

	DX12Texture2D::DX12Texture2D(Ref<Image> img)
	{
		PROFILE_SCOPE_FUNCTION();
		m_Path = "NONE";
		InitFromImage(img.get());
	}

	void DX12Texture2D::InitFromImage(Image* img)
	{
		m_Width = img->GetWidth();
		m_Height = img->GetHeight();
		m_Channel = img->GetChannel();

		bool hasAlpha = (m_Channel == 4);

		// -----------------------------------------
		// Create Defualt Resource
		// -----------------------------------------
		D3D12_RESOURCE_DESC texDesc = {};
		texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
		texDesc.Alignment = 0;
		texDesc.Width = (uint32_t)m_Width;
		texDesc.Height = (uint32_t)m_Height;
		texDesc.DepthOrArraySize = 1;
		texDesc.MipLevels = 1;
		texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		texDesc.SampleDesc.Count = 1;
		texDesc.SampleDesc.Quality = 0;
		texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
		texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;

		// Create Default Buffer
		DXCall(DX12Context::GetDevice()->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), // Format Default
			D3D12_HEAP_FLAG_SHARED,
			&texDesc,
			D3D12_RESOURCE_STATE_COMMON,
			nullptr,
			IID_PPV_ARGS(&m_Resource)
		));

		// Fetch Footprint
		D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
		UINT64  total_bytes = 0;
		DX12Context::GetDevice()->GetCopyableFootprints(&texDesc, 0, 1, 0, &footprint, nullptr, nullptr, &total_bytes);

		// -----------------------------------------
		// Create Upload Resource
		// -----------------------------------------
		// Allocate upload buffer
		D3D12_RESOURCE_DESC uploadTexDesc;
		memset(&uploadTexDesc, 0, sizeof(uploadTexDesc));
		uploadTexDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		uploadTexDesc.Width = total_bytes;
		uploadTexDesc.Height = 1;
		uploadTexDesc.DepthOrArraySize = 1;
		uploadTexDesc.MipLevels = 1;
		uploadTexDesc.SampleDesc.Count = 1;
		uploadTexDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

		//（7）创建上传堆
		D3D12_HEAP_PROPERTIES  uploadheap;
		memset(&uploadheap, 0, sizeof(uploadheap));
		uploadheap.Type = D3D12_HEAP_TYPE_UPLOAD;

		//（8）在上传堆上创建资源
		DXCall(DX12Context::GetDevice()->CreateCommittedResource(
			&uploadheap,
			D3D12_HEAP_FLAG_NONE,
			&uploadTexDesc,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&m_Uploader)
		));

		// Set defualt buffer to copy dest
		ID3D12GraphicsCommandList* cmdList = DX12Context::GetInFlightDXGraphicCommandList();
		cmdList->ResourceBarrier(1,
			&CD3DX12_RESOURCE_BARRIER::Transition(m_Resource.Get(),
				D3D12_RESOURCE_STATE_COMMON,
				D3D12_RESOURCE_STATE_COPY_DEST));

		D3D12_SUBRESOURCE_DATA subResourceData = {};
		subResourceData.pData = img->GetData();
		subResourceData.RowPitch = m_Width * 4;
		subResourceData.SlicePitch = total_bytes;

		// Copy Data from default buffer to Upload Buffer
		UpdateSubresources<1>(cmdList, m_Resource.Get(), m_Uploader.Get(), 0, 0, 1, &subResourceData);

		// Change the reosurce from COPY_DEST to GENERIC_READ
		cmdList->ResourceBarrier(1,
			&CD3DX12_RESOURCE_BARRIER::Transition(m_Resource.Get(),
				D3D12_RESOURCE_STATE_COPY_DEST,
				D3D12_RESOURCE_STATE_GENERIC_READ));
		// Create Descriptor
		Ref<DescriptorAllocator> srvAllocator = DX12Context::GetSrvDescriptorAllocator();
		m_DescriptorAllocation = srvAllocator->Allocate(1);


		D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
		srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
		srvDesc.Texture2D.MostDetailedMip = 0;
		srvDesc.Texture2D.MipLevels = 1;
		srvDesc.Texture2D.ResourceMinLODClamp = 0.0f;
		DX12Context::GetDevice()->CreateShaderResourceView(m_Resource.Get(), &srvDesc, m_DescriptorAllocation.GetDescriptorHandle());
	}

	DX12Texture2D::~DX12Texture2D()
	{
	}

	uint32_t DX12Texture2D::GetWidth() const
	{
		return m_Width;
	}

	uint32_t DX12Texture2D::GetHeight() const
	{
		return m_Height;
	}

	D3D12_CPU_DESCRIPTOR_HANDLE DX12Texture2D::GetSRVHandle()
	{
		return m_DescriptorAllocation.GetDescriptorHandle();
	}

	void DX12Texture2D::Bind(uint32_t slot) const
	{

	}

	void DX12Texture2D::RegisterImGui()
	{
		m_ImGuiAllocation = ImGuiLayerDX12::Get()->RegistShaderResource();

		DX12Context::GetDevice()->CreateShaderResourceView(
			m_Resource.Get(),
			nullptr,
			m_ImGuiAllocation.m_CpuHandle);
	}

	void* DX12Texture2D::GetImGuiHandle()
	{
		D3D12_GPU_DESCRIPTOR_HANDLE* handle = &m_ImGuiAllocation.m_GpuHandle;
		return *(void**)(handle); 
	}

}