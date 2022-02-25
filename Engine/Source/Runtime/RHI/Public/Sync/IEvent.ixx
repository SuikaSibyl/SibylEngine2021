module;
#include <cstdint>
export module RHI.IEvent;
import Core.SObject;

namespace SIByL
{
	namespace RHI
	{
		// ╔════════════════════╗
		// ║       Events       ║
		// ╚════════════════════╝
		// The idea of Event is to get some unrelated commands
		// in-between the “before” and “after” set of commands
		// 
		//  ╭╱────────────╲╮
		//  ╳    Example   ╳
		//  ╰╲────────────╱╯
		// 1.cmdDispatch
		// 2.cmdDispatch
		// 3.cmdSetEvent(event, srcStageMask = COMPUTE)
		// 4.cmdDispatch
		// 5.cmdWaitEvent(event, dstStageMask = COMPUTE)
		// 6.cmdDispatch
		// 7.cmdDispatch
		// The before set is {1,2}
		// The after set is {6,7}
		// {4} is not affected by any synchronization



		export class IEvent :public SObject
		{
		public:
			IEvent() = default;
			virtual ~IEvent() = default;
		};
	}
}
