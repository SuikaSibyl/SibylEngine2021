#pragma once

namespace SIByL
{
    namespace Graphic
    {
        struct ShaderReflection
        {
            unsigned int getEntryPointCount()
            {
                return 0;
                //return spReflection_getEntryPointCount((SlangReflection*)this);
            }
        };

        typedef ShaderReflection ProgramLayout;
    }
}