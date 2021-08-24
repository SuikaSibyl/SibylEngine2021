#pragma once

#include "Sibyl/Graphic/Texture/Texture.h"

namespace SIByL
{
	class DX12Texture2D :public Texture2D
	{
	public:
		DX12Texture2D(const std::string& path);
		virtual ~DX12Texture2D();

		virtual uint32_t GetWidth() const override;
		virtual uint32_t GetHeight() const override;

		virtual void Bind(uint32_t slot) const override;

	private:
		uint32_t m_Width;
		uint32_t m_Height;
		uint32_t m_ID;
		std::string m_Path;
	};
}