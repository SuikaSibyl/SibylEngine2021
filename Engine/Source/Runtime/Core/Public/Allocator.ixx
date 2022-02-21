module;
#include <cstddef>
#include <cstdint>
export module Core.Allocator;

namespace SIByL
{
	inline namespace Core
	{
        struct BlockHeader {
            // union-ed with data
            BlockHeader* pNext;
        };

        struct PageHeader {
            PageHeader* pNext;
            BlockHeader* Blocks() {
                return reinterpret_cast<BlockHeader*>(this + 1);
            }
        };

        export class Allocator
        {
        public:
            // debug patterns
            static const uint8_t PATTERN_ALIGN = 0xFC;
            static const uint8_t PATTERN_ALLOC = 0xFD;
            static const uint8_t PATTERN_FREE = 0xFE;

            Allocator();
            Allocator(size_t data_size, size_t page_size, size_t alignment);
            ~Allocator();

            // resets the allocator to a new configuration
            void reset(size_t data_size, size_t page_size, size_t alignment);

            // alloc and free blocks
            void* allocate();
            void  free(void* p);
            void  freeAll();

        private:
#if defined(_DEBUG)
            // fill a free page with debug patterns
            void fillFreePage(PageHeader* pPage);

            // fill a block with debug patterns
            void fillFreeBlock(BlockHeader* pBlock);

            // fill an allocated block with debug patterns
            void fillAllocatedBlock(BlockHeader* pBlock);
#endif

            // gets the next block
            BlockHeader* nextBlock(BlockHeader* pBlock);

            // the page list
            PageHeader* pPageList = nullptr;

            // the free block list
            BlockHeader* pFreeList;

            size_t      dataSize;
            size_t      pageSize;
            size_t      alignmentSize;
            size_t      blockSize;
            uint32_t    blocksPerPage;

            // statistics
            uint32_t    numPages;
            uint32_t    numBlocks;
            uint32_t    numFreeBlocks;

            // disable copy & assignment
            Allocator(const Allocator& clone);
            Allocator& operator=(const Allocator& rhs);
        };
	}
}