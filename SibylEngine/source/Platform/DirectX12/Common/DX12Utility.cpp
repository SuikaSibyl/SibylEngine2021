#include "SIByLpch.h"
#include "DX12Utility.h"
#include "DX12Context.h"

DxException::DxException(HRESULT hr, const std::wstring& functionName, const std::wstring& filename, int lineNumber) :
	ErrorCode(hr),
	FunctionName(functionName),
	Filename(filename),
	LineNumber(lineNumber)
{
}

namespace SIByL
{
	ComPtr<ID3D12Resource> CreateDefaultBuffer(UINT64 byteSize, const void* initData)
	{
		// Allocate upload buffer
		DX12UploadBuffer* uploadBuffer = DX12Context::GetUploadBuffer();
		DX12UploadBuffer::Allocation allocation = uploadBuffer->Allocate(byteSize, true);

		//Create Default Buffer as the target
		ComPtr<ID3D12Resource> defaultBuffer;
		DXCall(DX12Context::GetDevice()->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), // Type Default
			D3D12_HEAP_FLAG_NONE,
			&CD3DX12_RESOURCE_DESC::Buffer(byteSize),
			D3D12_RESOURCE_STATE_COMMON,
			nullptr,
			IID_PPV_ARGS(&defaultBuffer)));

		// Set defualt buffer to copy dest
		ID3D12GraphicsCommandList* cmdList = DX12Context::GetDXGraphicCommandList();
		cmdList->ResourceBarrier(1,
			&CD3DX12_RESOURCE_BARRIER::Transition(defaultBuffer.Get(),
				D3D12_RESOURCE_STATE_COMMON,
				D3D12_RESOURCE_STATE_COPY_DEST));

		// Copy the data from cpu to gpu
		D3D12_SUBRESOURCE_DATA subResourceData;
		subResourceData.pData = initData;
		subResourceData.RowPitch = byteSize;
		subResourceData.SlicePitch = subResourceData.RowPitch;
		// UpdateSubresources，Upload data from CPU to Upload Buffer,and then upload to Default Buffer
		// 1 is the sub-index of the largest subresource
		UpdateSubresources<1>(cmdList, defaultBuffer.Get(), allocation.Page, allocation.Offset, 0, 1, &subResourceData);

		// Change the reosurce from COPY_DEST to GENERIC_READ
		cmdList->ResourceBarrier(1,
			&CD3DX12_RESOURCE_BARRIER::Transition(defaultBuffer.Get(),
				D3D12_RESOURCE_STATE_COPY_DEST,
				D3D12_RESOURCE_STATE_GENERIC_READ));

		return defaultBuffer;
	}

	void UploadDefaultBuffer(ComPtr<ID3D12Resource> defaultBuffer, UINT64 byteSize, const void* initData)
	{
		//// Allocate upload buffer
		//D3D12_RESOURCE_DESC uploadTexDesc;
		//memset(&uploadTexDesc, 0, sizeof(uploadTexDesc));
		//uploadTexDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
		//uploadTexDesc.Width = byteSize;
		//uploadTexDesc.Height = 1;
		//uploadTexDesc.DepthOrArraySize = 1;
		//uploadTexDesc.MipLevels = 1;
		//uploadTexDesc.SampleDesc.Count = 1;
		//uploadTexDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

		////（7）创建上传堆
		//D3D12_HEAP_PROPERTIES  uploadheap;
		//memset(&uploadheap, 0, sizeof(uploadheap));
		//uploadheap.Type = D3D12_HEAP_TYPE_UPLOAD;

		////（8）在上传堆上创建资源
		//DXCall(DX12Context::GetDevice()->CreateCommittedResource(
		//	&uploadheap,
		//	D3D12_HEAP_FLAG_NONE,
		//	&uploadTexDesc,
		//	D3D12_RESOURCE_STATE_GENERIC_READ,
		//	nullptr,
		//	IID_PPV_ARGS(&Tex->UploadHeap)
		//));

		//// Set defualt buffer to copy dest
		//ID3D12GraphicsCommandList* cmdList = DX12Context::GetDXGraphicCommandList();
		//cmdList->ResourceBarrier(1,
		//	&CD3DX12_RESOURCE_BARRIER::Transition(defaultBuffer.Get(),
		//		D3D12_RESOURCE_STATE_COMMON,
		//		D3D12_RESOURCE_STATE_COPY_DEST));

		//D3D12_SUBRESOURCE_DATA subResourceData = {};
		//subResourceData.pData = initData;
		//subResourceData.RowPitch = byteSize;
		//subResourceData.SlicePitch = byteSize;

		//// Copy Data from default buffer to Upload Buffer
		//UpdateSubresources<1>(cmdList, defaultBuffer.Get(), allocation.Page, allocation.Offset, 0, 1, &subResourceData);

		//// Change the reosurce from COPY_DEST to GENERIC_READ
		//cmdList->ResourceBarrier(1,
		//	&CD3DX12_RESOURCE_BARRIER::Transition(defaultBuffer.Get(),
		//		D3D12_RESOURCE_STATE_COPY_DEST,
		//		D3D12_RESOURCE_STATE_GENERIC_READ));
	}
}