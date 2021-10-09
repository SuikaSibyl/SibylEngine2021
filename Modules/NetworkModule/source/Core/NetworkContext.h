#pragma once

namespace SIByLNetwork
{
	struct IP
	{
		unsigned char seg[4];
	};

	class NetworkContext
	{
	public:
		static void Init();
		static IP GetLocalIP();

	private:
		static IP localIP;
	};
}