#pragma once

namespace SIByL
{
	class SRenderPipe
	{
	public:
		
		const std::string& GetName();

	protected:
		std::string m_Name;
	};
}