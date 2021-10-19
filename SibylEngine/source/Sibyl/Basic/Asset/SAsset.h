#pragma once

#include <filesystem>

namespace SIByL
{
	struct SAssetBuffer
	{
		unsigned int BufferElementCount;
		unsigned int BufferByteSize;
		unsigned int BufferExtraHeadSize = 0;

		char* Buffer = nullptr;
		char* BufferHead = nullptr;
		char* BufferExtraHead = nullptr;

		template <class T>
		void LoadFromVector(const std::vector<T> data);

		template <class T>
		void LoadToVector(std::vector<T>& data);

		template <class T>
		void SetExtraHead(const T& data);

		template <class T>
		void LoadExtraHead(T& data);

		void ReleaseBuffer();
		void ReleaseExtraHeadBuffer();
		void Serialize(std::ofstream& ofstream);
		void Deserialize(std::ifstream& ifstream);
	};

	template <class T>
	void SAssetBuffer::LoadFromVector(const std::vector<T> data)
	{
		BufferElementCount = data.size();
		BufferByteSize = data.size() * sizeof(T);
		if (Buffer != nullptr) ReleaseBuffer();
		Buffer = new char[BufferByteSize];
		memcpy(Buffer, data.data(), BufferByteSize);
	}

	template <class T>
	void SAssetBuffer::LoadToVector(std::vector<T>& data)
	{
		data.clear();
		data.resize(BufferElementCount);
		memcpy(data.data(), Buffer, BufferByteSize);
	}

	template <class T>
	void SAssetBuffer::SetExtraHead(const T& data)
	{
		BufferExtraHeadSize = sizeof(T);
		if (BufferExtraHead != nullptr) ReleaseExtraHeadBuffer();
		BufferExtraHead = new char[BufferExtraHeadSize];
		memcpy(BufferExtraHead, &data, BufferExtraHeadSize);
	}

	template <class T>
	void SAssetBuffer::LoadExtraHead(T& data)
	{
		memcpy(&data, BufferExtraHead, BufferExtraHeadSize);
	}

	class SAsset
	{
	public:
		virtual ~SAsset() = default;

		void SetSavePath(const std::string& path);
		void SetSavePath(const std::filesystem::path& path);
		std::string GetSavePath();

		void ReleaseBuffers();

		virtual void Serialize();
		virtual bool Deserialize();

	protected:
		std::vector<SAssetBuffer> Buffers;
		std::filesystem::path SavePath;
	};
}