module;
#include <cstddef>
#include <cstdint>
#include <cstring>
module Core.Allocator;
import Core.Assert;

#ifndef ALIGN
#define ALIGN(x, a) (((x) + ((a) - 1)) & ~((a) - 1))
#endif

namespace SIByL::Core
{
    Allocator::Allocator()
        : pPageList(nullptr), pFreeList(nullptr)
    {}

    Allocator::Allocator(size_t data_size, size_t page_size, size_t alignment)
        : pPageList(nullptr), pFreeList(nullptr)
    {
        reset(data_size, page_size, alignment);
    }

    Allocator::~Allocator()
    {
        freeAll();
    }

    void Allocator::reset(size_t data_size, size_t page_size, size_t alignment)
    {
        freeAll();

        dataSize = data_size;
        pageSize = page_size;

        size_t minimal_size = (sizeof(BlockHeader) > dataSize) ? sizeof(BlockHeader) : dataSize;
        // this magic only works when alignment is 2^n, which should general be the case
        // because most CPU/GPU also requires the aligment be in 2^n
        // but still we use a assert to guarantee it
#if defined(_DEBUG)
        SE_CORE_ASSERT(alignment > 0 && ((alignment & (alignment - 1))) == 0, "allocator reset get wrong minimalzie");
#endif
        blockSize = ALIGN(minimal_size, alignment);

        alignmentSize = blockSize - minimal_size;

        blocksPerPage = (pageSize - sizeof(PageHeader)) / blockSize;
    }

    void* Allocator::allocate()
    {
        if (!pFreeList) {
            // allocate a new page
            PageHeader* pNewPage = reinterpret_cast<PageHeader*>(new uint8_t[pageSize]);

            ++numPages;
            numBlocks += blocksPerPage;
            numFreeBlocks += blocksPerPage;

#if defined(_DEBUG)
            fillFreePage(pNewPage);
#endif

            if (pPageList) {
                pNewPage->pNext = pPageList;
            }

            pPageList = pNewPage;

            BlockHeader* pBlock = pNewPage->Blocks();
            // link each block in the page
            for (uint32_t i = 0; i < blocksPerPage - 1; i++) {
                pBlock->pNext = nextBlock(pBlock);
                pBlock = nextBlock(pBlock);
            }
            pBlock->pNext = nullptr;

            pFreeList = pNewPage->Blocks();
        }

        BlockHeader* freeBlock = pFreeList;
        pFreeList = pFreeList->pNext;
        --numFreeBlocks;

#if defined(_DEBUG)
        fillAllocatedBlock(freeBlock);
#endif

        return reinterpret_cast<void*>(freeBlock);
    }

    void Allocator::free(void* p)
    {
        BlockHeader* block = reinterpret_cast<BlockHeader*>(p);

#if defined(_DEBUG)
        fillFreeBlock(block);
#endif

        block->pNext = pFreeList;
        pFreeList = block;
        ++numFreeBlocks;
    }

    void Allocator::freeAll()
    {
        PageHeader* pPage = pPageList;
        while (pPage) {
            PageHeader* _p = pPage;
            pPage = pPage->pNext;

            delete[] reinterpret_cast<uint8_t*>(_p);
        }

        pPageList = nullptr;
        pFreeList = nullptr;

        numPages = 0;
        numBlocks = 0;
        numFreeBlocks = 0;
    }

#if defined(_DEBUG)
    void Allocator::fillFreePage(PageHeader* pPage)
    {
        // page header
        pPage->pNext = nullptr;

        // blocks
        BlockHeader* pBlock = pPage->Blocks();
        for (uint32_t i = 0; i < blocksPerPage; i++)
        {
            fillFreeBlock(pBlock);
            pBlock = nextBlock(pBlock);
        }
    }

    void Allocator::fillFreeBlock(BlockHeader* pBlock)
    {
        // block header + data
        std::memset(pBlock, PATTERN_FREE, blockSize - alignmentSize);

        // alignment
        std::memset(reinterpret_cast<uint8_t*>(pBlock) + blockSize - alignmentSize,
            PATTERN_ALIGN, alignmentSize);
    }

    void Allocator::fillAllocatedBlock(BlockHeader* pBlock)
    {
        // block header + data
        std::memset(pBlock, PATTERN_ALLOC, blockSize - alignmentSize);

        // alignment
        std::memset(reinterpret_cast<uint8_t*>(pBlock) + blockSize - alignmentSize,
            PATTERN_ALIGN, alignmentSize);
    }
#endif
    BlockHeader* Allocator::nextBlock(BlockHeader* pBlock)
    {
        return reinterpret_cast<BlockHeader*>(reinterpret_cast<uint8_t*>(pBlock) + blockSize);
    }
}