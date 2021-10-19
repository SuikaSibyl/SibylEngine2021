#include "SIByLpch.h"
#include "SAsset.h"

namespace SIByL
{
	void SAssetBuffer::ReleaseBuffer()
	{
		delete[] Buffer;
	}

	void SAssetBuffer::ReleaseExtraHeadBuffer()
	{
		delete[] BufferExtraHead;
	}

	void SAssetBuffer::Serialize(std::ofstream& ofstream)
	{
		BufferHead = new char[3 * sizeof(unsigned int)];
		unsigned int Head[3] = { BufferElementCount ,BufferByteSize, BufferExtraHeadSize };
		memcpy(BufferHead, Head, 3 * sizeof(unsigned int));
		ofstream.write(BufferHead, 3 * sizeof(unsigned int));
		if (BufferExtraHeadSize != 0)
		{
			ofstream.write(BufferExtraHead, BufferExtraHeadSize);
		}
		ofstream.write(Buffer, BufferByteSize);
		delete[] BufferHead;
	}

	void SAssetBuffer::Deserialize(std::ifstream& ifstream)
	{
		BufferHead = new char[3 * sizeof(unsigned int)];
		unsigned int Head[3] = { 0 ,0, 0 };
		ifstream.read(BufferHead, 3 * sizeof(unsigned int));
		memcpy(Head, BufferHead, 3 * sizeof(unsigned int));
		BufferElementCount = Head[0];
		BufferByteSize = Head[1];
		BufferExtraHeadSize = Head[2];
		if (BufferExtraHeadSize != 0)
		{
			BufferExtraHead = new char[BufferExtraHeadSize];
			ifstream.read(BufferExtraHead, BufferExtraHeadSize);
		}
		Buffer = new char[BufferByteSize];
		ifstream.read(Buffer, BufferByteSize);

		delete[] BufferHead;
	}

	void SAsset::SetSavePath(const std::string& path)
	{
		SavePath = path;
	}

	void SAsset::SetSavePath(const std::filesystem::path& path)
	{
		SavePath = path;
	}

	std::string SAsset::GetSavePath()
	{
		return SavePath.string();
	}

	void SAsset::Serialize()
	{
		std::ofstream writeFile;
		writeFile.open(SavePath, std::ios::out | std::ios::binary);
		SIByL_CORE_ASSERT(writeFile, "Error Open Asset Save Path");
		
		char* BufferHead = new char[sizeof(unsigned int)];
		unsigned int Head[1] = { Buffers.size() };
		memcpy(BufferHead, Head, sizeof(unsigned int));
		writeFile.write(BufferHead, sizeof(unsigned int));

		// Write All the buffers
		for (auto& buffer : Buffers)
		{
			buffer.Serialize(writeFile);
		}
		
		writeFile.close();
		ReleaseBuffers();
	}

	void SAsset::ReleaseBuffers()
	{
		for (auto& buffer : Buffers)
		{
			buffer.ReleaseBuffer();
		}
		Buffers.clear();
	}

	bool SAsset::Deserialize()
	{
		std::ifstream readFile;
		readFile.open(SavePath, std::ios::in | std::ios::binary);
		if (!readFile)
		{
			return false;
		}

		char* BufferHead = new char[sizeof(unsigned int)];
		unsigned int Head[1] = { 0 };
		readFile.read(BufferHead, sizeof(unsigned int));
		memcpy(Head, BufferHead, sizeof(unsigned int));

		Buffers.resize(Head[0]);
		for (int i = 0; i < Head[0]; i++)
		{
			Buffers[i].Deserialize(readFile);
		}

		delete[] BufferHead;
		return true;
	}
}