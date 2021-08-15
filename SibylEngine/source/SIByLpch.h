#pragma once

#include <iostream>
#include <memory>
#include <cstdint>
#include <utility>
#include <algorithm>
#include <functional>
#include <new>

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <codecvt>
#include <comdef.h>

#include <vector>
#include <deque>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "Sibyl/Core/Log.h"

#ifdef SIBYL_PLATFORM_WINDOWS

#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "D3D12.lib")
#pragma comment(lib, "dxgi.lib")

#include <Windows.h>
#include <wincodec.h>
#include <windowsx.h>
#include <wrl.h>
#include <dxgi1_4.h>

#include <d3d12.h>

#include <d3dcompiler.h>
#include <DirectXMath.h>
#include <DirectXPackedVector.h>
#include <DirectXColors.h>
#include <DirectXCollision.h>
#include <Platform/DirectX12/include/d3dx12.h>

using namespace Microsoft::WRL;
using namespace DirectX;

#endif // SIBYL_PLATFORM_WINDOWS