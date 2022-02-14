module;
#include <slang.h>
#include <slang-com-ptr.h>
export module RHI.SlangUtility;
import Core.Log;

// Many Slang API functions return detailed diagnostic information
// (error messages, warnings, etc.) as a "blob" of data, or return
// a null blob pointer instead if there were no issues.
//
// For convenience, we define a subroutine that will dump the information
// in a diagnostic blob if one is produced, and skip it otherwise.
//
export inline void diagnoseIfNeeded(slang::IBlob* diagnosticsBlob)
{
    if (diagnosticsBlob != nullptr)
    {
        SE_CORE_ERROR("SLANG :: {0}", (const char*)diagnosticsBlob->getBufferPointer());
    }
}