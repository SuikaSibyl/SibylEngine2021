#include "SShaderTypes.h"

namespace SIByL
{
	namespace Shader
	{
        //Kind getKind()
        //{
        //    return (Kind)spReflectionType_GetKind((SlangReflectionType*)this);
        //}

        //// only useful if `getKind() == Kind::Struct`
        //unsigned int getFieldCount()
        //{
        //    return spReflectionType_GetFieldCount((SlangReflectionType*)this);
        //}

        //VariableReflection* getFieldByIndex(unsigned int index)
        //{
        //    return (VariableReflection*)spReflectionType_GetFieldByIndex((SlangReflectionType*)this, index);
        //}

        //bool isArray() { return getKind() == TypeReflection::Kind::Array; }

        //TypeReflection* unwrapArray()
        //{
        //    TypeReflection* type = this;
        //    while (type->isArray())
        //    {
        //        type = type->getElementType();
        //    }
        //    return type;
        //}

        //// only useful if `getKind() == Kind::Array`
        //size_t getElementCount()
        //{
        //    return spReflectionType_GetElementCount((SlangReflectionType*)this);
        //}

        //size_t getTotalArrayElementCount()
        //{
        //    if (!isArray()) return 0;
        //    size_t result = 1;
        //    TypeReflection* type = this;
        //    for (;;)
        //    {
        //        if (!type->isArray())
        //            return result;

        //        result *= type->getElementCount();
        //        type = type->getElementType();
        //    }
        //}

        //TypeReflection* getElementType()
        //{
        //    return (TypeReflection*)spReflectionType_GetElementType((SlangReflectionType*)this);
        //}

        //unsigned getRowCount()
        //{
        //    return spReflectionType_GetRowCount((SlangReflectionType*)this);
        //}

        //unsigned getColumnCount()
        //{
        //    return spReflectionType_GetColumnCount((SlangReflectionType*)this);
        //}

        //ScalarType getScalarType()
        //{
        //    return (ScalarType)spReflectionType_GetScalarType((SlangReflectionType*)this);
        //}

        //TypeReflection* getResourceResultType()
        //{
        //    return (TypeReflection*)spReflectionType_GetResourceResultType((SlangReflectionType*)this);
        //}

        //SlangResourceShape getResourceShape()
        //{
        //    return spReflectionType_GetResourceShape((SlangReflectionType*)this);
        //}

        //SlangResourceAccess getResourceAccess()
        //{
        //    return spReflectionType_GetResourceAccess((SlangReflectionType*)this);
        //}

        //char const* getName()
        //{
        //    return spReflectionType_GetName((SlangReflectionType*)this);
        //}

        //unsigned int getUserAttributeCount()
        //{
        //    return spReflectionType_GetUserAttributeCount((SlangReflectionType*)this);
        //}
        //UserAttribute* getUserAttributeByIndex(unsigned int index)
        //{
        //    return (UserAttribute*)spReflectionType_GetUserAttribute((SlangReflectionType*)this, index);
        //}
        //UserAttribute* findUserAttributeByName(char const* name)
        //{
        //    return (UserAttribute*)spReflectionType_FindUserAttributeByName((SlangReflectionType*)this, name);
        //}
	}
}