#include "SIByLpch.h"
#include "Drawing.h"

namespace SIByL
{
	RenderQueueRange RenderQueueRange::all = { 0, 10000 };
	RenderQueueRange RenderQueueRange::opaque = { 0, 3000 };
	RenderQueueRange RenderQueueRange::transparent = { 3000, 6000 };
}