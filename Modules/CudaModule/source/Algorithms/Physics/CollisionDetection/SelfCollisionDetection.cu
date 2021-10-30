#pragma once

#include "CudaModulePCH.h"
#include "SelfCollisionDetection.h"
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <queue>

#define CUDACalll checkCudaErrors
#define THREADS_PER_BLOCK 512

namespace SIByL
{
	namespace CUDA
	{
		typedef unsigned int MortonType;
		typedef unsigned int IndexType;

		struct TriangleVertices
		{
			float v1_x, v1_y, v1_z, v2_x, v2_y, v2_z, v3_x, v3_y, v3_z;
		};

		struct TriangleIndices
		{
			int p1, p2, p3;
		};

		struct BoundingBox
		{
			float xmin;
			float xmax;
			float ymin;
			float ymax;
			float zmin;
			float zmax;
		};

		struct TrianglesSoA
		{
			TriangleVertices* vertices;
			TriangleIndices* indices;
		};

		enum NodeType
		{
			LEAFNODE,
			INTERNALNODE,
		};

		struct Node
		{
			unsigned int Index;
			int ParentIndex;
			NodeType Type;
		};

		struct ChildrenInfo
		{
			NodeType childTypeLeft;
			NodeType childTypeRight;
			unsigned int childIndxLeft;
			unsigned int childIndxRight;
		};

		struct RangeInfo
		{
			unsigned int leftMost;
			unsigned int rightMost;
		};

		struct InternalNodesSoA
		{
			Node* Nodes;
			BoundingBox* BoundingBoxes;
			ChildrenInfo* ChildrenInfos;
			RangeInfo* RangeInfos;
		};

		struct TriangleInfoAoF
		{
			unsigned int PrimitiveIdx;
			TriangleVertices Vertices;
			TriangleIndices Indices;
		};

		struct LeafNodesSoA
		{
			Node* Nodes;
			BoundingBox* BoundingBoxes;

			unsigned int* PrimitiveIdx;
			TriangleVertices* Vertices;
			TriangleIndices* Indices;
		};

#pragma region KERNELS
		__global__ void InitTriangleKernel(unsigned int triangleNum,
			float* d_vertices, IndexType* d_indices,	// Input
			TriangleInfoAoF* triangleAoF, BoundingBox* d_WorldSpaceBBoxes) // Output
		{
			int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx >= triangleNum) return;

			// Init primitive
			triangleAoF[idx].PrimitiveIdx = idx;

			// Init indices
			unsigned int idx1 = d_indices[3 * idx + 0];
			unsigned int idx2 = d_indices[3 * idx + 1];
			unsigned int idx3 = d_indices[3 * idx + 2];

			triangleAoF[idx].Indices.p1 = idx1;
			triangleAoF[idx].Indices.p2 = idx2;
			triangleAoF[idx].Indices.p3 = idx3;

			// Init vertices
			float v1x = d_vertices[3 * idx1 + 0];
			float v1y = d_vertices[3 * idx1 + 1];
			float v1z = d_vertices[3 * idx1 + 2];

			float v2x = d_vertices[3 * idx2 + 0];
			float v2y = d_vertices[3 * idx2 + 1];
			float v2z = d_vertices[3 * idx2 + 2];

			float v3x = d_vertices[3 * idx3 + 0];
			float v3y = d_vertices[3 * idx3 + 1];
			float v3z = d_vertices[3 * idx3 + 2];

			triangleAoF[idx].Vertices.v1_x = v1x;
			triangleAoF[idx].Vertices.v1_y = v1y;
			triangleAoF[idx].Vertices.v1_z = v1z;
			triangleAoF[idx].Vertices.v2_x = v2x;
			triangleAoF[idx].Vertices.v2_y = v2y;
			triangleAoF[idx].Vertices.v2_z = v2z;
			triangleAoF[idx].Vertices.v3_x = v3x;
			triangleAoF[idx].Vertices.v3_y = v3y;
			triangleAoF[idx].Vertices.v3_z = v3z;

			// Init BBoxes
			d_WorldSpaceBBoxes[idx].xmin = (fmin(fmin(v1x, v2x), v3x));
			d_WorldSpaceBBoxes[idx].xmax = (fmax(fmax(v1x, v2x), v3x));
			d_WorldSpaceBBoxes[idx].ymin = (fmin(fmin(v1y, v2y), v3y));
			d_WorldSpaceBBoxes[idx].ymax = (fmax(fmax(v1y, v2y), v3y));
			d_WorldSpaceBBoxes[idx].zmin = (fmin(fmin(v1z, v2z), v3z));
			d_WorldSpaceBBoxes[idx].zmax = (fmax(fmax(v1z, v2z), v3z));
		}

		__device__ MortonType expandBits(MortonType v)
		{
			v = (v * 0x00010001u) & 0xFF0000FFu;
			v = (v * 0x00000101u) & 0x0F00F00Fu;
			v = (v * 0x00000011u) & 0xC30C30C3u;
			v = (v * 0x00000005u) & 0x49249249u;

			return v;
		}

		__global__ void MortonCalcKernel(unsigned int triangleCount, 
			BoundingBox* worldSpaceBBoxes, BoundingBox sceneBBox, float scale,
			MortonType* keys)
		{
			int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx >= triangleCount) return;

			worldSpaceBBoxes[idx].xmin = fmax(0.01f, fmin(0.99f, (worldSpaceBBoxes[idx].xmin - sceneBBox.xmin) * scale));
			worldSpaceBBoxes[idx].xmax = fmax(0.01f, fmin(0.99f, (worldSpaceBBoxes[idx].xmax - sceneBBox.xmin) * scale));
			worldSpaceBBoxes[idx].ymin = fmax(0.01f, fmin(0.99f, (worldSpaceBBoxes[idx].ymin - sceneBBox.ymin) * scale));
			worldSpaceBBoxes[idx].ymax = fmax(0.01f, fmin(0.99f, (worldSpaceBBoxes[idx].ymax - sceneBBox.ymin) * scale));
			worldSpaceBBoxes[idx].zmin = fmax(0.01f, fmin(0.99f, (worldSpaceBBoxes[idx].zmin - sceneBBox.zmin) * scale));
			worldSpaceBBoxes[idx].zmax = fmax(0.01f, fmin(0.99f, (worldSpaceBBoxes[idx].zmax - sceneBBox.zmin) * scale));

			float x, y, z;
			x = (worldSpaceBBoxes[idx].xmin + worldSpaceBBoxes[idx].xmax) / 2;
			y = (worldSpaceBBoxes[idx].ymin + worldSpaceBBoxes[idx].ymax) / 2;
			z = (worldSpaceBBoxes[idx].zmin + worldSpaceBBoxes[idx].zmax) / 2;

			x = x * 1023.0f;
			y = y * 1023.0f;
			z = z * 1023.0f;

			MortonType xx = expandBits((MortonType)(x));
			MortonType yy = expandBits((MortonType)(y));
			MortonType zz = expandBits((MortonType)(z));

			keys[idx] = xx * 4 + yy * 2 + zz;
		}

		__global__ void AssignLeafNodesKernel(unsigned int triangleCount,
			TriangleInfoAoF* triangleAoF, BoundingBox* uniformSpaceBBoxes,
			LeafNodesSoA leafNodes)
		{
			int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx >= triangleCount) return;

			leafNodes.Nodes[idx].Type = LEAFNODE;
			leafNodes.Nodes[idx].Index = idx;

			unsigned int primitiveIndex = triangleAoF[idx].PrimitiveIdx;
			leafNodes.PrimitiveIdx[idx] = primitiveIndex;
			leafNodes.BoundingBoxes[idx] = uniformSpaceBBoxes[primitiveIndex];
			leafNodes.Vertices[idx] = triangleAoF[idx].Vertices;
			leafNodes.Indices[idx] = triangleAoF[idx].Indices;
		}

#pragma region BVHMortonBuildHelper
		__device__ int delta(MortonType* sortedMortonCodes, unsigned int x, unsigned int y, unsigned int numObjects)
		{
			if (x >= 0 && x <= numObjects - 1 && y >= 0 && y <= numObjects - 1)
			{
				int delta = __clz(sortedMortonCodes[x] ^ sortedMortonCodes[y]);
				if (sortedMortonCodes[x] == sortedMortonCodes[y]) delta += __clz(x ^ y);
				return delta;
			}
			return -1;
		}

		__device__ int sign(int x)
		{
			return (x > 0) - (x < 0);
		}

		__device__ int2 determineRange(MortonType* sortedMortonCodes, int numObjects, int idx)
		{
			int dleft = delta(sortedMortonCodes, idx, idx - 1, numObjects);
			int dright = delta(sortedMortonCodes, idx, idx + 1, numObjects);

			int d = sign(dright - dleft);
			int dmin = (d == 1) ? dleft : dright;
			int lmax = 2;
			while (delta(sortedMortonCodes, idx, idx + lmax * d, numObjects) > dmin)
				lmax = lmax * 2;
			int l = 0;
			for (int t = lmax / 2; t >= 1; t /= 2)
			{
				if (delta(sortedMortonCodes, idx, idx + (l + t) * d, numObjects) > dmin)
					l += t;
			}

			int j = idx + l * d;
			int2 range;
			range.x = min(idx, j);
			range.y = max(idx, j);

			return range;
		}

		__device__ int findSplit(MortonType* sortedMortonCodes,
			int first,
			int last,
			int numObjects)
		{
			int commonPrefix = delta(sortedMortonCodes, first, last, numObjects);
			int split = first; // initial guess
			int step = last - first;
			do
			{
				step = (step + 1) >> 1; // exponential decrease
				int newSplit = split + step; // proposed new position

				if (newSplit < last)
				{
					int splitPrefix = delta(sortedMortonCodes, first, newSplit, numObjects);
					if (splitPrefix > commonPrefix)
						split = newSplit; // accept proposal
				}
			} while (step > 1);

			return split;
		}
#pragma endregion

		__global__ void AssignInternalNodesKernel(unsigned int triangleCount,
			MortonType* sortedMortonCodes, LeafNodesSoA leafNodes,
			InternalNodesSoA internalNodes)
		{
			int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx >= triangleCount - 1) return;

			// Find the range first
			int2 range = determineRange(sortedMortonCodes, triangleCount, idx);
			int first = range.x;
			int last = range.y;

			internalNodes.RangeInfos[idx].leftMost = first;
			internalNodes.RangeInfos[idx].rightMost = last;

			// Find the split position then
			int split = findSplit(sortedMortonCodes, first, last, triangleCount);

			// Select childA.
			Node* childA;
			int childAIdx;
			NodeType childAType;
			if (split == first)
			{
				childA = &leafNodes.Nodes[split];
				childAIdx = split;
				childAType = LEAFNODE;
			}
			else
			{
				childA = &internalNodes.Nodes[split];
				childAIdx = split;
				childAType = INTERNALNODE;
			}
			// Select childB.
			Node* childB;
			int childBIdx;
			NodeType childBType;
			if (split + 1 == last)
			{
				childB = &leafNodes.Nodes[split + 1];
				childBIdx = split + 1;
				childBType = LEAFNODE;
			}
			else
			{
				childB = &internalNodes.Nodes[split + 1];
				childBIdx = split + 1;
				childBType = INTERNALNODE;
			}

			// Record parent-child relationships.
			internalNodes.Nodes[idx].Index = idx;
			internalNodes.Nodes[idx].Type = INTERNALNODE;
			internalNodes.ChildrenInfos[idx].childIndxLeft = childAIdx;
			internalNodes.ChildrenInfos[idx].childTypeLeft = childAType;
			internalNodes.ChildrenInfos[idx].childIndxRight = childBIdx;
			internalNodes.ChildrenInfos[idx].childTypeRight = childBType;

			if (idx == 0) internalNodes.Nodes[idx].ParentIndex = -1;

			childA->ParentIndex = idx;
			childB->ParentIndex = idx;
		}

		struct CombineAABB {
			__host__ __device__
			BoundingBox operator()(const BoundingBox& left, const BoundingBox& right) const {
				BoundingBox res;
				res.xmin = fmin(left.xmin, right.xmin);
				res.xmax = fmax(left.xmax, right.xmax);
				res.ymin = fmin(left.ymin, right.ymin);
				res.ymax = fmax(left.ymax, right.ymax);
				res.zmin = fmin(left.zmin, right.zmin);
				res.zmax = fmax(left.zmax, right.zmax);
				return res;
			}
		};

		__host__ __device__
		BoundingBox CombineAABBFunc(const BoundingBox& left, const BoundingBox& right)
		{
			BoundingBox res;
			res.xmin = fmin(left.xmin, right.xmin);
			res.xmax = fmax(left.xmax, right.xmax);
			res.ymin = fmin(left.ymin, right.ymin);
			res.ymax = fmax(left.ymax, right.ymax);
			res.zmin = fmin(left.zmin, right.zmin);
			res.zmax = fmax(left.zmax, right.zmax);
			return res;
		}

		__global__ void AssignInternalNodeBBoxKernel(unsigned int triangleCount,
			LeafNodesSoA leafNodes, InternalNodesSoA internalNodes, int* atom)
		{
			int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx >= triangleCount) return;

			Node* node = &leafNodes.Nodes[idx];
			unsigned int parentIdx = node->ParentIndex;

			while (parentIdx != -1 && atomicAdd(&atom[parentIdx], 1) == 1)
			{
				BoundingBox left = internalNodes.ChildrenInfos[parentIdx].childTypeLeft == INTERNALNODE ?
					internalNodes.BoundingBoxes[internalNodes.ChildrenInfos[parentIdx].childIndxLeft] :
					leafNodes.BoundingBoxes[internalNodes.ChildrenInfos[parentIdx].childIndxLeft];

				BoundingBox right = internalNodes.ChildrenInfos[parentIdx].childTypeRight == INTERNALNODE ?
					internalNodes.BoundingBoxes[internalNodes.ChildrenInfos[parentIdx].childIndxRight] :
					leafNodes.BoundingBoxes[internalNodes.ChildrenInfos[parentIdx].childIndxRight];

				internalNodes.BoundingBoxes[parentIdx] = CombineAABBFunc(left, right);
				__threadfence();

				parentIdx = internalNodes.Nodes[parentIdx].ParentIndex;
			}

		}

#define ALLOC_OFFSET 80

		__host__ __device__ bool AABBTest(const BoundingBox& a, const BoundingBox& b)
		{
			return !(a.xmin > b.xmax || b.xmin > a.xmax ||
				a.ymin > b.ymax || b.ymin > a.ymax ||
				a.zmin > b.zmax || b.zmin > a.zmax);
		}

		__device__ bool CullNeightbor(const TriangleIndices& a, const TriangleIndices& b)
		{
			return !((
				a.p1 == b.p1 || a.p1 == b.p2 || a.p1 == b.p3 ||
				a.p2 == b.p1 || a.p2 == b.p2 || a.p2 == b.p3 ||
				a.p3 == b.p1 || a.p3 == b.p2 || a.p3 == b.p3));
		}

		__device__ void traverseIterative(
			int* list, int offset,
			LeafNodesSoA leafNodes, InternalNodesSoA internalNodes,
			BoundingBox queryBBox, unsigned int queryIdx, TriangleIndices queryIndices)
		{
			// Allocate traversal stack from thread-local memory,
			// and push NULL to indicate that there are no postponed nodes.
			unsigned int stack[64];
			unsigned int* stackPtr = stack;
			*stackPtr++ = -1; // push

			// Traverse nodes starting from the root
			unsigned int presentIdx = 0;
			do
			{
				unsigned int childrenLeftIdx = internalNodes.ChildrenInfos[presentIdx].childIndxLeft;
				unsigned int childrenRightIdx = internalNodes.ChildrenInfos[presentIdx].childIndxRight;
				NodeType childrenLeftType = internalNodes.ChildrenInfos[presentIdx].childTypeLeft;
				NodeType childrenRightType = internalNodes.ChildrenInfos[presentIdx].childTypeRight;

				BoundingBox& bboxLeft = childrenLeftType == INTERNALNODE ?
					internalNodes.BoundingBoxes[childrenLeftIdx] : leafNodes.BoundingBoxes[childrenLeftIdx];
				BoundingBox& bboxRight = childrenRightType == INTERNALNODE ?
					internalNodes.BoundingBoxes[childrenRightIdx] : leafNodes.BoundingBoxes[childrenRightIdx];

				bool overlapL = (AABBTest(queryBBox, bboxLeft));
				bool overlapR = (AABBTest(queryBBox, bboxRight));

				if (((childrenLeftType == INTERNALNODE && internalNodes.RangeInfos[childrenLeftIdx].rightMost <= queryIdx) ||
					(childrenLeftType == LEAFNODE && (queryIdx >= childrenLeftIdx))) && overlapL)
					overlapL = false;

				if (((childrenRightType == INTERNALNODE && internalNodes.RangeInfos[childrenRightIdx].rightMost <= queryIdx) ||
					(childrenRightType == LEAFNODE && (queryIdx >= childrenRightIdx))) && overlapR)
					overlapR = false;

				// Query overlaps a leaf node => report collision.
				if (overlapL && childrenLeftType == LEAFNODE && CullNeightbor(queryIndices, leafNodes.Indices[childrenLeftIdx]))
				{
					list[offset++] = queryIdx;
					list[offset++] = childrenLeftIdx;
				}
				if (overlapR && childrenRightType == LEAFNODE && CullNeightbor(queryIndices, leafNodes.Indices[childrenRightIdx]))
				{
					list[offset++] = queryIdx;
					list[offset++] = childrenRightIdx;
				}

				// Query overlaps an internal node => traverse.
				bool traverseL = (overlapL && childrenLeftType == INTERNALNODE);
				bool traverseR = (overlapR && childrenRightType == INTERNALNODE);

				if (!traverseL && !traverseR)
				{
					presentIdx = *--stackPtr; // pop
				}
				else
				{
					presentIdx = (traverseL) ? childrenLeftIdx : childrenRightIdx;
					if (traverseL && traverseR)
					{
						*stackPtr++ = childrenRightIdx; // push
					}
				}
			} while (presentIdx != -1);
		}

		__global__ void FindPotentialCollisionsKernel(unsigned int triangleCount,
			LeafNodesSoA leafNodes, InternalNodesSoA internalNodes,
			int* list)
		{
			int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx < triangleCount)
			{
				traverseIterative(
					list, idx * ALLOC_OFFSET,
					leafNodes, internalNodes,
					leafNodes.BoundingBoxes[idx], idx, leafNodes.Indices[idx]);
			}
		}
#pragma region TriangleCDCheck
		__device__ __host__
			inline int project3(const float3& ax,
				const float3& p1, const float3& p2, const float3& p3)
		{
			float P1 = dot(ax, p1);
			float P2 = dot(ax, p2);
			float P3 = dot(ax, p3);

			float mx1 = max(max(P1, P2), P3);
			float mn1 = min(min(P1, P2), P3);

			if (mn1 > 0) return 0;
			if (0 > mx1) return 0;
			return 1;
		}

		__device__ __host__
			inline int project6(float3& ax,
				float3& p1, float3& p2, float3& p3,
				float3& q1, float3& q2, float3& q3)
		{
			float P1 = dot(ax, p1);
			float P2 = dot(ax, p2);
			float P3 = dot(ax, p3);
			float Q1 = dot(ax, q1);
			float Q2 = dot(ax, q2);
			float Q3 = dot(ax, q3);

			float mx1 = max(max(P1, P2), P3);
			float mn1 = min(min(P1, P2), P3);
			float mx2 = max(max(Q1, Q2), Q3);
			float mn2 = min(min(Q1, Q2), Q3);

			if (mn1 > mx2) return 0;
			if (mn2 > mx1) return 0;
			return 1;
		}

		__device__ __host__
			bool tri_contact(float3 P1, float3 P2, float3 P3, float3 Q1, float3 Q2, float3 Q3)
		{
			float3 p1 = make_float3(0, 0, 0);
			float3 p2 = P2 - P1;
			float3 p3 = P3 - P1;
			float3 q1 = Q1 - P1;
			float3 q2 = Q2 - P1;
			float3 q3 = Q3 - P1;

			float3 e1 = p2 - p1;
			float3 e2 = p3 - p2;
			float3 e3 = p1 - p3;

			float3 f1 = q2 - q1;
			float3 f2 = q3 - q2;
			float3 f3 = q1 - q3;

			float3 n1 = cross(e1, e2);
			float3 m1 = cross(f1, f2);

			float3 g1 = cross(e1, n1);
			float3 g2 = cross(e2, n1);
			float3 g3 = cross(e3, n1);

			float3 h1 = cross(f1, m1);
			float3 h2 = cross(f2, m1);
			float3 h3 = cross(f3, m1);

			float3 ef11 = cross(e1, f1);
			float3 ef12 = cross(e1, f2);
			float3 ef13 = cross(e1, f3);
			float3 ef21 = cross(e2, f1);
			float3 ef22 = cross(e2, f2);
			float3 ef23 = cross(e2, f3);
			float3 ef31 = cross(e3, f1);
			float3 ef32 = cross(e3, f2);
			float3 ef33 = cross(e3, f3);

			// now begin the series of tests
			if (!project3(n1, q1, q2, q3)) return false;
			if (!project3(m1, -q1, p2 - q1, p3 - q1)) return false;

			if (!project6(ef11, p1, p2, p3, q1, q2, q3)) return false;
			if (!project6(ef12, p1, p2, p3, q1, q2, q3)) return false;
			if (!project6(ef13, p1, p2, p3, q1, q2, q3)) return false;
			if (!project6(ef21, p1, p2, p3, q1, q2, q3)) return false;
			if (!project6(ef22, p1, p2, p3, q1, q2, q3)) return false;
			if (!project6(ef23, p1, p2, p3, q1, q2, q3)) return false;
			if (!project6(ef31, p1, p2, p3, q1, q2, q3)) return false;
			if (!project6(ef32, p1, p2, p3, q1, q2, q3)) return false;
			if (!project6(ef33, p1, p2, p3, q1, q2, q3)) return false;
			if (!project6(g1, p1, p2, p3, q1, q2, q3)) return false;
			if (!project6(g2, p1, p2, p3, q1, q2, q3)) return false;
			if (!project6(g3, p1, p2, p3, q1, q2, q3)) return false;
			if (!project6(h1, p1, p2, p3, q1, q2, q3)) return false;
			if (!project6(h2, p1, p2, p3, q1, q2, q3)) return false;
			if (!project6(h3, p1, p2, p3, q1, q2, q3)) return false;

			return true;
		}
#pragma endregion

		__global__ void TrianglePairCollisionDetectionKernel(unsigned int pairSize,
			LeafNodesSoA leafNodes, int* input,
			int* output)
		{
			int idx = threadIdx.x + blockDim.x * blockIdx.x;
			if (idx < pairSize)
			{
				TriangleVertices* left = &leafNodes.Vertices[input[idx * 2 + 0]];
				TriangleVertices* right = &leafNodes.Vertices[input[idx * 2 + 1]];

				if (tri_contact(
					make_float3(left->v1_x, left->v1_y, left->v1_z),
					make_float3(left->v2_x, left->v2_y, left->v2_z),
					make_float3(left->v3_x, left->v3_y, left->v3_z),
					make_float3(right->v1_x, right->v1_y, right->v1_z),
					make_float3(right->v2_x, right->v2_y, right->v2_z),
					make_float3(right->v3_x, right->v3_y, right->v3_z)))
				{
					unsigned int idx1 = leafNodes.PrimitiveIdx[input[idx * 2 + 0]];
					unsigned int idx2 = leafNodes.PrimitiveIdx[input[idx * 2 + 1]];
					output[idx * 2 + 0] = idx1 < idx2 ? idx1 : idx2;
					output[idx * 2 + 1] = idx1 > idx2 ? idx1 : idx2;
				}
			}
		}

		struct notminusone
		{
			__host__ __device__
				bool operator()(const int x)
			{
				return x != -1;
			}
		};

#pragma endregion

#define BeginEvent(x)	cudaEvent_t start_##x, stop_##x; \
						cudaEventCreate(&start_##x);\
						cudaEventCreate(&stop_##x); \
						cudaDeviceSynchronize(); \
						cudaEventRecord(start_##x);
#define EndEvent(x, str) cudaEventRecord(stop_##x);\
						cudaEventSynchronize(stop_##x);\
						float milliseconds_##x = 0;\
						cudaEventElapsedTime(&milliseconds_##x, start_##x, stop_##x);\
						cudaEventDestroy(start_##x);\
						cudaEventDestroy(stop_##x);\
						printf(str, milliseconds_##x);

		void SelfCollisionDetection::FindCollision(const std::vector<float>& vertices, const std::vector<IndexType>& indices, 
			std::vector<std::pair<int, int>>& collided_pairs)
		{
			clock_t start, end;
			start = std::clock();

			// Number of triangles
			unsigned int triangleNum = indices.size() / 3;

			// Copy vertex & index data to device
			float* d_vertices;
			IndexType* d_indices;
			CUDACalll(cudaMalloc((void**)&(d_vertices), sizeof(float) * triangleNum * 3 * 3));
			CUDACalll(cudaMalloc((void**)&(d_indices), sizeof(IndexType) * triangleNum * 3));
			CUDACalll(cudaMemcpy(d_vertices, vertices.data(), sizeof(float) * vertices.size(), cudaMemcpyHostToDevice));
			CUDACalll(cudaMemcpy(d_indices, indices.data(), sizeof(IndexType) * triangleNum * 3, cudaMemcpyHostToDevice));

			// Init Traingle info & boundingboxes
			TriangleInfoAoF* d_triangleAoF;
			BoundingBox* d_WorldSpaceBBoxes;
			CUDACalll(cudaMalloc((void**)&(d_triangleAoF), sizeof(TriangleInfoAoF) * triangleNum));
			CUDACalll(cudaMalloc((void**)&(d_WorldSpaceBBoxes), sizeof(BoundingBox) * triangleNum));
			InitTriangleKernel << <BLOCK_SIZE(triangleNum, 32), 32 >> >
				(triangleNum, d_vertices, d_indices, // Input
					d_triangleAoF, d_WorldSpaceBBoxes); // Output

			// Find the world space Scene Bounding Box, calc the scale size
			thrust::device_ptr<BoundingBox> d_WorldSpaceBBoxes_ptr(d_WorldSpaceBBoxes);
			BoundingBox InitBox = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
			BoundingBox SceneBox = thrust::reduce(d_WorldSpaceBBoxes_ptr, d_WorldSpaceBBoxes_ptr + triangleNum, InitBox, CombineAABB());
			float _scale = fmin(fmin(1.0f / (SceneBox.xmax - SceneBox.xmin), 1.0f / (SceneBox.ymax - SceneBox.ymin)), 1.0f / (SceneBox.zmax - SceneBox.zmin));

			// Calculate Morton Code for each bounding box
			MortonType* d_MortonCodes;
			CUDACalll(cudaMalloc((void**)&(d_MortonCodes), sizeof(MortonType) * triangleNum));
			MortonCalcKernel << <BLOCK_SIZE(triangleNum, 32), 32 >> >
				(triangleNum, d_WorldSpaceBBoxes, SceneBox, _scale, d_MortonCodes);

			// Sort Triangles by Morton Codes
			thrust::device_ptr<MortonType> d_MortonCodes_ptr(d_MortonCodes);
			thrust::device_ptr<TriangleInfoAoF>  d_triangleAoF_ptr(d_triangleAoF);
			thrust::sort_by_key(d_MortonCodes_ptr, d_MortonCodes_ptr + triangleNum, d_triangleAoF_ptr);

			// Init Leaf Nodes
			LeafNodesSoA d_LeafNodes;
			CUDACalll(cudaMalloc((void**)&(d_LeafNodes.Nodes), sizeof(Node) * triangleNum));
			CUDACalll(cudaMalloc((void**)&(d_LeafNodes.BoundingBoxes), sizeof(BoundingBox) * triangleNum));
			CUDACalll(cudaMalloc((void**)&(d_LeafNodes.PrimitiveIdx), sizeof(unsigned int) * triangleNum));
			CUDACalll(cudaMalloc((void**)&(d_LeafNodes.Vertices), sizeof(TriangleVertices) * triangleNum));
			CUDACalll(cudaMalloc((void**)&(d_LeafNodes.Indices), sizeof(TriangleIndices) * triangleNum));
			AssignLeafNodesKernel << <BLOCK_SIZE(triangleNum, 32), 32 >> >
				(triangleNum, d_triangleAoF, d_WorldSpaceBBoxes, d_LeafNodes);

			// Assign Internal Nodes
			InternalNodesSoA d_InternalNodes;
			CUDACalll(cudaMalloc((void**)&(d_InternalNodes.Nodes), sizeof(Node) * (triangleNum - 1)));
			CUDACalll(cudaMalloc((void**)&(d_InternalNodes.BoundingBoxes), sizeof(BoundingBox) * (triangleNum - 1)));
			CUDACalll(cudaMalloc((void**)&(d_InternalNodes.ChildrenInfos), sizeof(ChildrenInfo) * (triangleNum - 1)));
			CUDACalll(cudaMalloc((void**)&(d_InternalNodes.RangeInfos), sizeof(RangeInfo) * (triangleNum - 1)));
			AssignInternalNodesKernel << <BLOCK_SIZE((triangleNum - 1), 512), 512 >> >
				(triangleNum, d_MortonCodes, d_LeafNodes, d_InternalNodes);

			// Assign Internal Nodes AABB BBoxes
			int* atom;
			CUDACalll(cudaMalloc((void**)&atom, sizeof(int) * (triangleNum - 1)));
			CUDACalll(cudaMemset(atom, 0, sizeof(int) * (triangleNum - 1)));
			AssignInternalNodeBBoxKernel << <BLOCK_SIZE(triangleNum, 32), 32 >> >
				(triangleNum, d_LeafNodes, d_InternalNodes, atom);

			// Find possible pairs
			int* d_possible_pair_list;
			CUDACalll(cudaMalloc((void**)&d_possible_pair_list, triangleNum * ALLOC_OFFSET * sizeof(int)));
			CUDACalll(cudaMemset(d_possible_pair_list, -1, triangleNum * ALLOC_OFFSET * sizeof(int)));
			FindPotentialCollisionsKernel << <BLOCK_SIZE(triangleNum - 1, 256), 256 >> >
				(triangleNum, d_LeafNodes, d_InternalNodes, d_possible_pair_list);

			// Filter those -1 pairs
			int* d_possible_pair_list_filtered;
			CUDACalll(cudaMalloc((void**)&d_possible_pair_list_filtered, triangleNum * ALLOC_OFFSET * sizeof(int)));
			thrust::device_ptr<int> d_possible_pair_list_ptr(d_possible_pair_list);
			thrust::device_ptr<int> d_possible_pair_list_filtered_ptr(d_possible_pair_list_filtered);
			thrust::device_ptr<int> d_possible_pair_list_ptr_end = 
				thrust::copy_if(d_possible_pair_list_ptr, d_possible_pair_list_ptr + triangleNum * ALLOC_OFFSET, d_possible_pair_list_filtered_ptr, notminusone());
			unsigned int num_filter_pair_list = thrust::raw_pointer_cast(d_possible_pair_list_ptr_end) - d_possible_pair_list_filtered;
			num_filter_pair_list /= 2;

			// Final Check possible pairs
			int* d_collided_pair_list;
			CUDACalll(cudaMalloc((void**)&d_collided_pair_list, num_filter_pair_list * 2 * sizeof(int)));
			CUDACalll(cudaMemset(d_collided_pair_list, -1, num_filter_pair_list * 2 * sizeof(int)));
			TrianglePairCollisionDetectionKernel << <BLOCK_SIZE(triangleNum - 1, 32), 32 >> >
				(num_filter_pair_list, d_LeafNodes, d_possible_pair_list_filtered, d_collided_pair_list);

			// Filter those -1 pairs
			int* d_filtered_collided_pairs;
			CUDACalll(cudaMalloc((void**)&d_filtered_collided_pairs, num_filter_pair_list * 2 * sizeof(int)));
			thrust::device_ptr<int> d_collided_pair_ptr(d_collided_pair_list);
			thrust::device_ptr<int> d_filtered_collided_ptr(d_filtered_collided_pairs);
			thrust::device_ptr<int> final_filter_res = thrust::copy_if(d_collided_pair_ptr, d_collided_pair_ptr + num_filter_pair_list * 2, d_filtered_collided_ptr, notminusone());
			unsigned int num_final_pair_nums = thrust::raw_pointer_cast(final_filter_res) - d_filtered_collided_pairs;
			num_final_pair_nums /= 2;

			// Copy back results to CPU
			collided_pairs.resize(num_final_pair_nums);
			CUDACalll(cudaMemcpy(collided_pairs.data(), d_filtered_collided_pairs, sizeof(int)* num_final_pair_nums * 2, cudaMemcpyDeviceToHost));
			cudaThreadSynchronize();


			end = clock();
			double endtime2 = (double)(end - start) / CLOCKS_PER_SEC;
			std::cout << "Time period: " << endtime2 << ", pair nums:" << num_final_pair_nums << std::endl;
		}
	}
}