module;
#include <cstdint>
export module RHI.IMemoryBarrier;
import RHI.IEnum;
import RHI.ICommandQueue;
import RHI.ITexture;

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
			virtual ~IMemoryBarrier() = default;
		};

        // ╔════════════════════════════════╗
        // ║     Buffer Memory Barrier      ║
        // ╚════════════════════════════════╝
        // It is quite similar to Memory Barrier
        // Memory availability and visibility are restricted to a specific buffer.
        
        export class IBufferMemoryBarrier
        {
        public:
            IBufferMemoryBarrier() = default;
            virtual ~IBufferMemoryBarrier() = default;
        };

        // ╔═══════════════════════════════╗
        // ║     Image Memory Barrier      ║
        // ╚═══════════════════════════════╝
        // Beyond the memory barrier,
        // Image Memory Barrier also take cares about layout change,
        // The layout transition happens in-between the make available and make visible stages
        // The layout transition itself is considered a read/write operation,
        // the memory for image must be available before transition takes place,
        // After a layout transition, the memory is automatically made available.
        //
        // Could think of the layout transition
        // as some kind of in-place data munging which happens in L2 cache
        //
        // It can also be used to transfer queue family ownership when SHARING_MODE_EXCLUSIVE is used

        export class IImageMemoryBarrier
        {
        public:
            IImageMemoryBarrier() = default;
            virtual ~IImageMemoryBarrier() = default;
        };

		// ╔════════════════════╗
		// ║    Access Flags    ║
		// ╚════════════════════╝
        // Access Flags describe the access need for barrier.
        // In memory barrier, Access Flags are combined with Stage Flags.
        // 
        // Warning: do not use AccessMask!=0 with TOP_OF_PIPE/BOTTOM_OF_PIPE
        // Because these stages do not perform memory accesses,
        // they are purely used for execution barriers

		export enum class AccessFlagBits :uint32_t
		{
            INDIRECT_COMMAND_READ_BIT = 0x00000001,
            INDEX_READ_BIT = 0x00000002,
            VERTEX_ATTRIBUTE_READ_BIT = 0x00000004,
            UNIFORM_READ_BIT = 0x00000008,
            INPUT_ATTACHMENT_READ_BIT = 0x00000010,
            SHADER_READ_BIT = 0x00000020,
            SHADER_WRITE_BIT = 0x00000040,
            COLOR_ATTACHMENT_READ_BIT = 0x00000080,
            COLOR_ATTACHMENT_WRITE_BIT = 0x00000100,
            DEPTH_STENCIL_ATTACHMENT_READ_BIT = 0x00000200,
            DEPTH_STENCIL_ATTACHMENT_WRITE_BIT = 0x00000400,
            TRANSFER_READ_BIT = 0x00000800,
            TRANSFER_WRITE_BIT = 0x00001000,
            HOST_READ_BIT = 0x00002000,
            HOST_WRITE_BIT = 0x00004000,
            MEMORY_READ_BIT = 0x00008000,
            MEMORY_WRITE_BIT = 0x00010000,
            TRANSFORM_FEEDBACK_WRITE_BIT = 0x02000000,
            TRANSFORM_FEEDBACK_COUNTER_READ_BIT = 0x04000000,
            TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT = 0x08000000,
            CONDITIONAL_RENDERING_READ_BIT = 0x00100000,
            COLOR_ATTACHMENT_READ_NONCOHERENT_BIT = 0x00080000,
            ACCELERATION_STRUCTURE_READ_BIT = 0x00200000,
            ACCELERATION_STRUCTURE_WRITE_BIT = 0x00400000,
            FRAGMENT_DENSITY_MAP_READ_BIT = 0x01000000,
            FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT = 0x00800000,
            COMMAND_PREPROCESS_READ_BIT = 0x00020000,
            COMMAND_PREPROCESS_WRITE_BIT = 0x00040000,
            NONE = 0,
		};
        export using AccessFlags = uint32_t;

        // ╔══════════════════════════════╗
        // ║     Memory Barrier Desc      ║
        // ╚══════════════════════════════╝
        export struct MemoryBarrierDesc
        {
            // memory barrier mask
            AccessFlags srcAccessMask;
            AccessFlags dstAccessMask;
        };

        // ╔════════════════════════════════════╗
        // ║    Buffer Memory Barrier Desc      ║
        // ╚════════════════════════════════════╝
        export struct BufferMemoryBarrierDesc
        {

        };

        // ╔═══════════════════════════════════╗
        // ║    Image Memory Barrier Desc      ║
        // ╚═══════════════════════════════════╝
        export struct ImageMemoryBarrierDesc
        {
            // specify image object
            ITexture* image;
            ImageSubresourceRange subresourceRange;
            // memory barrier mask
            AccessFlags srcAccessMask;
            AccessFlags dstAccessMask;
            // only if layout transition is need
            ImageLayout oldLayout;
            ImageLayout newLayout;
            // only if queue transition is need
            ICommandQueue* srcQueue = nullptr;
            ICommandQueue* dstQueue = nullptr;
        };
	}
}