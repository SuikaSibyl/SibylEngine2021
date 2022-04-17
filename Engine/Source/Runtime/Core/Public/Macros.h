#pragma once

#include <functional>
#define BIND_EVENT_FN(x) std::bind(&x, this, std::placeholders::_1)

#ifdef _DEBUG
#define PROFILE_SCOPE(name) InstrumentationTimer timer##__LINE__(name);
#define PROFILE_SCOPE_FUNCTION() PROFILE_SCOPE(__FUNCSIG__)
#else
#define PROFILE_SCOPE(name)  
#define PROFILE_SCOPE_FUNCTION()  
#endif