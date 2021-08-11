#pragma once

#include <iostream>
#include <memory>
#include <cstdint>
#include <utility>
#include <algorithm>
#include <functional>

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <codecvt>

#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "Sibyl/Core/Log.h"

#ifdef SIBYL_PLATFORM_WINDOWS

#include <Windows.h>
#include <windowsx.h>
#include <wrl.h>
#include <dxgi1_4.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
#include <DirectXPackedVector.h>
#include <DirectXColors.h>
#include <DirectXCollision.h>

using namespace Microsoft::WRL;

#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "D3D12.lib")
#pragma comment(lib, "dxgi.lib")

#endif // SIBYL_PLATFORM_WINDOWS