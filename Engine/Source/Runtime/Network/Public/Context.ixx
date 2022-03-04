export module Network.Context;

namespace SIByL::Network
{
	export struct IP
	{
		unsigned char seg[4];
	};

	export class Context
	{
	public:
		Context();
		static auto instance() noexcept -> Context*;
		auto getLocalIP() noexcept -> IP;

	private:
		IP localIP;
	};
}