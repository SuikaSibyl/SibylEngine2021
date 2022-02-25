export module RHI.IMemoryBarrier;

namespace SIByL
{
	namespace RHI
	{
		// ╔══════════════════════════╗
		// ║      Memory Barrier      ║
		// ╚══════════════════════════╝
		// Memory barrier is a structure specifying a global memory barrier
		// A global memory barrier deals with access to any resource, 
		// and it’s the simplest form of a memory barrier. 
		// 
		// Description includes:
		//  ► AccessFlags - srcAccessMask
		//  ► AccessFlags - dstAccessMask


		export class IMemoryBarrier
		{
		public:
			IMemoryBarrier() = default;
			virtual ~IMemoryBarrier() = 0;
		};

	}
}