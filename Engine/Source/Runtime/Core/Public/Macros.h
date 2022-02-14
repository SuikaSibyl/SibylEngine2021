#pragma once

#include <functional>
#define BIND_EVENT_FN(x) std::bind(&x, this, std::placeholders::_1)