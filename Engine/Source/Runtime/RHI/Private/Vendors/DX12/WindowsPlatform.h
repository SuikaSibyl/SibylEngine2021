#pragma once

#ifndef SAFE_RELEASE
	#define SAFE_RELEASE(p_var) \
		if(p_var)				\
		{						\
			p_var->Release();	\
			p_var = NULL;		\
		}						
#endif // !SAFE_RELEASE

#pragma comment(lib, "dxgi.lib")

#include <Windows.h>
#include <dxgi1_6.h>
#include <WinNls.h>
#include <d3d12.h>