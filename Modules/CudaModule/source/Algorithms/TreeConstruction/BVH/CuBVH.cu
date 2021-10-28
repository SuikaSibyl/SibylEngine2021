#pragma once

#include "CudaModulePCH.h"
#include "CuBVH.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <queue>

namespace SIByL
{
	typedef unsigned int MortonType;

	struct MortonRec {
		float x, y, z;
		float xx, yy, zz;
		MortonType ex, ey, ez;
		MortonType m;
		unsigned int index;
	};

	struct Triangle
	{
		float v1_x, v1_y, v1_z, v2_x, v2_y, v2_z, v3_x, v3_y, v3_z;
		int p1, p2, p3;
	};

	void CuBVH::LoadData(const std::vector<float>& vertices, const std::vector<uint32_t>& indices, unsigned int stepsize)
	{
		std::vector<Triangle> triangles;
		// step 9: v1_x, v1_y, v1_z, v2_x, v2_y, v2_z, v3_x, v3_y, v3_z
		std::vector<float> boundingboxes;
		// step 6: xmin, xmax. ymin, ymax, zmin, zmax

		for (unsigned int i = 0; i < indices.size(); i += 3)
		{
			float v1x = vertices[stepsize * indices[i + 0] + 0];
			float v1y = vertices[stepsize * indices[i + 0] + 1];
			float v1z = vertices[stepsize * indices[i + 0] + 2];

			float v2x = vertices[stepsize * indices[i + 1] + 0];
			float v2y = vertices[stepsize * indices[i + 1] + 1];
			float v2z = vertices[stepsize * indices[i + 1] + 2];

			float v3x = vertices[stepsize * indices[i + 2] + 0];
			float v3y = vertices[stepsize * indices[i + 2] + 1];
			float v3z = vertices[stepsize * indices[i + 2] + 2];

			Triangle triangle;
			triangle.p1 = indices[i + 0];
			triangle.p2 = indices[i + 1];
			triangle.p3 = indices[i + 2];
			triangle.v1_x = v1x;
			triangle.v1_y = v1y;
			triangle.v1_z = v1z;
			triangle.v2_x = v2x;
			triangle.v2_y = v2y;
			triangle.v2_z = v2z;
			triangle.v3_x = v3x;
			triangle.v3_y = v3y;
			triangle.v3_z = v3z;
			triangles.emplace_back(triangle);

			boundingboxes.push_back(fmin(fmin(v1x, v2x), v3x));
			boundingboxes.push_back(fmax(fmax(v1x, v2x), v3x));
			boundingboxes.push_back(fmin(fmin(v1y, v2y), v3y));
			boundingboxes.push_back(fmax(fmax(v1y, v2y), v3y));
			boundingboxes.push_back(fmin(fmin(v1z, v2z), v3z));
			boundingboxes.push_back(fmax(fmax(v1z, v2z), v3z));

		}
		// must be bounded to unit cube
		float bounds[6] = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
		for (int i = 0; i < boundingboxes.size(); i += 6)
		{
			bounds[0] = fmin(bounds[0], boundingboxes[i + 0]);
			bounds[1] = fmax(bounds[1], boundingboxes[i + 1]);
			bounds[2] = fmin(bounds[2], boundingboxes[i + 2]);
			bounds[3] = fmax(bounds[3], boundingboxes[i + 3]);
			bounds[4] = fmin(bounds[4], boundingboxes[i + 4]);
			bounds[5] = fmax(bounds[5], boundingboxes[i + 5]);
		}

		float _scale = fmin(fmin(1.0f / (bounds[1] - bounds[0]), 1.0f / (bounds[3] - bounds[2])), 1.0f / (bounds[5] - bounds[4]));
		for (int i = 0; i < boundingboxes.size(); i += 6)
		{
			boundingboxes[i + 0] = fmax(0.01, fmin(0.99, (boundingboxes[i + 0] - bounds[0]) * _scale));
			boundingboxes[i + 1] = fmax(0.01, fmin(0.99, (boundingboxes[i + 1] - bounds[0]) * _scale));
			boundingboxes[i + 2] = fmax(0.01, fmin(0.99, (boundingboxes[i + 2] - bounds[2]) * _scale));
			boundingboxes[i + 3] = fmax(0.01, fmin(0.99, (boundingboxes[i + 3] - bounds[2]) * _scale));
			boundingboxes[i + 4] = fmax(0.01, fmin(0.99, (boundingboxes[i + 4] - bounds[4]) * _scale));
			boundingboxes[i + 5] = fmax(0.01, fmin(0.99, (boundingboxes[i + 5] - bounds[4]) * _scale));
		}

		BuildBVH(triangles, boundingboxes);
	}

	__device__ MortonType expandBits(MortonType v)
	{
		v = (v * 0x00010001u) & 0xFF0000FFu;
		v = (v * 0x00000101u) & 0x0F00F00Fu;
		v = (v * 0x00000011u) & 0xC30C30C3u;
		v = (v * 0x00000005u) & 0x49249249u;

		return v;
	}

	__global__ void Morton3DKernel(unsigned int triangleCount, MortonType* keys, const float* boundingboxes, MortonRec* records)
	{
		int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= triangleCount) return;

		float x, y, z;
		x = (boundingboxes[6 * idx + 0] + boundingboxes[6 * idx + 1]) / 2;
		y = (boundingboxes[6 * idx + 2] + boundingboxes[6 * idx + 3]) / 2;
		z = (boundingboxes[6 * idx + 4] + boundingboxes[6 * idx + 5]) / 2;

		records[idx].x = x;
		records[idx].y = y;
		records[idx].z = z;

		x = x * 1023.0f;
		y = y * 1023.0f;
		z = z * 1023.0f;

		records[idx].xx = x;
		records[idx].yy = y;
		records[idx].zz = z;
		
		MortonType xx = expandBits((MortonType)(x));
		MortonType yy = expandBits((MortonType)(y));
		MortonType zz = expandBits((MortonType)(z));
		keys[idx] = xx * 4 + yy * 2 + zz;

		records[idx].ex = xx;
		records[idx].ey = yy;
		records[idx].ez = zz;
		records[idx].m = keys[idx];
		records[idx].index = idx;
	}

#define THREADS_PER_BLOCK 512
	enum NodeType
	{
		LEAFNODE,
		INTERNALNODE,
	};

	struct BBox
	{
		float xmin;
		float xmax;
		float ymin;
		float ymax;
		float zmin;
		float zmax;
	};

	struct Node
	{
		unsigned int Index;
		NodeType Type;
		BBox BoundingBox;

		int ParentIndex;

		__device__ void setParent(int idx)
		{
			ParentIndex = idx;
		}
		__device__ unsigned int getParent() { return ParentIndex; }
	};

	struct LeafNode : public Node
	{
		unsigned int primitiveIdx;
		float v1_x, v1_y, v1_z, v2_x, v2_y, v2_z, v3_x, v3_y, v3_z;
		int p1, p2, p3;
	};

	struct InternalNode : public Node
	{
		NodeType childTypeLeft;
		NodeType childTypeRight;
		unsigned int childIndxLeft;
		unsigned int childIndxRight;

		__device__ void SetChildLeft(NodeType type, unsigned int idx)
		{
			childTypeLeft = type;
			childIndxLeft = idx;
		}

		__device__ void SetChildRight(NodeType type, unsigned int idx)
		{
			childTypeRight = type;
			childIndxRight = idx;
		}
	};

	__global__ void AssignLeafNodesKernel(unsigned int triangleCount, LeafNode* leafNodes, MortonRec* mortonRec, const float* boundingboxes, Triangle* d_triangles)
	{
		int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= triangleCount) return;

		leafNodes[idx].Type = LEAFNODE;
		leafNodes[idx].Index = idx;
		unsigned int primitiveIndex = mortonRec[idx].index;
		leafNodes[idx].primitiveIdx = primitiveIndex;
		leafNodes[idx].BoundingBox.xmin = boundingboxes[6 * primitiveIndex + 0];
		leafNodes[idx].BoundingBox.xmax = boundingboxes[6 * primitiveIndex + 1];
		leafNodes[idx].BoundingBox.ymin = boundingboxes[6 * primitiveIndex + 2];
		leafNodes[idx].BoundingBox.ymax = boundingboxes[6 * primitiveIndex + 3];
		leafNodes[idx].BoundingBox.zmin = boundingboxes[6 * primitiveIndex + 4];
		leafNodes[idx].BoundingBox.zmax = boundingboxes[6 * primitiveIndex + 5];

		leafNodes[idx].v1_x = d_triangles[primitiveIndex].v1_x;
		leafNodes[idx].v1_y = d_triangles[primitiveIndex].v1_y;
		leafNodes[idx].v1_z = d_triangles[primitiveIndex].v1_z;
		leafNodes[idx].v2_x = d_triangles[primitiveIndex].v2_x;
		leafNodes[idx].v2_y = d_triangles[primitiveIndex].v2_y;
		leafNodes[idx].v2_z = d_triangles[primitiveIndex].v2_z;
		leafNodes[idx].v3_x = d_triangles[primitiveIndex].v3_x;
		leafNodes[idx].v3_y = d_triangles[primitiveIndex].v3_y;
		leafNodes[idx].v3_z = d_triangles[primitiveIndex].v3_z;

		leafNodes[idx].p1 = d_triangles[primitiveIndex].p1;
		leafNodes[idx].p2 = d_triangles[primitiveIndex].p2;
		leafNodes[idx].p3 = d_triangles[primitiveIndex].p3;
	}

	__device__ int delta(MortonType* sortedMortonCodes, unsigned int x, unsigned int y, unsigned int numObjects)
	{
		if (x >= 0 && x <= numObjects - 1 && y >= 0 && y <= numObjects - 1)
		{
			int delta = __clz(sortedMortonCodes[x] ^ sortedMortonCodes[y]);
			if (sortedMortonCodes[x] == sortedMortonCodes[y]) delta += __clz(x^y);
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

		//printf("Index: %d    dleft: %d, dright: %d  \n", idx, dleft, dright);

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
		// Identical Morton codes => split the range in the middle.
		//MortonType firstCode = sortedMortonCodes[first];
		//MortonType lastCode = sortedMortonCodes[last];

		//if (firstCode == lastCode)
		//	return (first + last) >> 1;

		// Calculate the number of highest bits that are the same
		// for all objects, using the count-leading-zeros intrinsic.

//	// 	int commonPrefix = __clz(firstCode ^ lastCode);
//#if HASH_64
//		int commonPrefix = __clzll(firstCode ^ lastCode);
//#else
//		int commonPrefix = __clz(firstCode ^ lastCode);
//#endif
		int commonPrefix = delta(sortedMortonCodes, first, last, numObjects);
		// Use binary search to find where the next bit differs.
		// Specifically, we are looking for the highest object that
		// shares more than commonPrefix bits with the first one.

		int split = first; // initial guess
		int step = last - first;

		do
		{
			step = (step + 1) >> 1; // exponential decrease
			int newSplit = split + step; // proposed new position

			if (newSplit < last)
			{
				//MortonType splitCode = sortedMortonCodes[newSplit];
				// 			int splitPrefix = __clz(firstCode ^ splitCode);
				int splitPrefix = delta(sortedMortonCodes, first, newSplit, numObjects);
				if (splitPrefix > commonPrefix)
					split = newSplit; // accept proposal
			}
		} while (step > 1);

		return split;
	}

	__global__ void AssignInternalNodesKernel(unsigned int triangleCount, MortonType* sortedMortonCodes, LeafNode* leafNodes, InternalNode* internalNodes, MortonRec* mortonRec)
	{
		int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= triangleCount - 1) return;

		int2 range = determineRange(sortedMortonCodes, triangleCount, idx);
		int first = range.x;
		int last = range.y;

		// Determine where to split the range.
		int split = findSplit(sortedMortonCodes, first, last, triangleCount);

		// Select childA.
		Node* childA;
		int childAIdx;
		NodeType childAType;
		if (split == first)
		{
			childA = &leafNodes[split];
			childAIdx = split;
			childAType = LEAFNODE;
		}
		else
		{
			childA = &internalNodes[split];
			childAIdx = split;
			childAType = INTERNALNODE;
		}
		// Select childB.
		Node* childB;
		int childBIdx;
		NodeType childBType;
		if (split + 1 == last)
		{
			childB = &leafNodes[split + 1];
			childBIdx = split + 1;
			childBType = LEAFNODE;
		}
		else
		{
			childB = &internalNodes[split + 1];
			childBIdx = split + 1;
			childBType = INTERNALNODE;
		}

		// Record parent-child relationships.
		internalNodes[idx].Index = idx;
		internalNodes[idx].Type = INTERNALNODE;
		internalNodes[idx].SetChildLeft(childAType, childAIdx);
		internalNodes[idx].SetChildRight(childBType, childBIdx);
		if (idx == 0) internalNodes[idx].ParentIndex = -1;

		childA->setParent(idx);
		childB->setParent(idx);
	}

	__global__ void InternalNodeBBoxKernel(int triangleCount, int* atom, InternalNode* internalNodes, LeafNode* leafNodes)
	{
		int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= triangleCount) return;

		Node* ptr = &leafNodes[idx];
		unsigned int parentIdx = ptr->getParent();
		InternalNode* parent = &internalNodes[parentIdx];

		while (atomicCAS(&atom[parentIdx], 0, 1) == 1)
		{
			BBox buf;
			float left, right;

			left = parent->childTypeLeft == INTERNALNODE ? internalNodes[parent->childIndxLeft].BoundingBox.xmax : leafNodes[parent->childIndxLeft].BoundingBox.xmax;
			right = parent->childTypeRight == INTERNALNODE ? internalNodes[parent->childIndxRight].BoundingBox.xmax : leafNodes[parent->childIndxRight].BoundingBox.xmax;
			buf.xmax = fmax(left, right);
			left = parent->childTypeLeft == INTERNALNODE ? internalNodes[parent->childIndxLeft].BoundingBox.xmin : leafNodes[parent->childIndxLeft].BoundingBox.xmin;
			right = parent->childTypeRight == INTERNALNODE ? internalNodes[parent->childIndxRight].BoundingBox.xmin : leafNodes[parent->childIndxRight].BoundingBox.xmin;
			buf.xmin = fmin(left, right);

			left = parent->childTypeLeft == INTERNALNODE ? internalNodes[parent->childIndxLeft].BoundingBox.ymax : leafNodes[parent->childIndxLeft].BoundingBox.ymax;
			right = parent->childTypeRight == INTERNALNODE ? internalNodes[parent->childIndxRight].BoundingBox.ymax : leafNodes[parent->childIndxRight].BoundingBox.ymax;
			buf.ymax = fmax(left, right);
			left = parent->childTypeLeft == INTERNALNODE ? internalNodes[parent->childIndxLeft].BoundingBox.ymin : leafNodes[parent->childIndxLeft].BoundingBox.ymin;
			right = parent->childTypeRight == INTERNALNODE ? internalNodes[parent->childIndxRight].BoundingBox.ymin : leafNodes[parent->childIndxRight].BoundingBox.ymin;
			buf.ymin = fmin(left, right);

			left = parent->childTypeLeft == INTERNALNODE ? internalNodes[parent->childIndxLeft].BoundingBox.zmax : leafNodes[parent->childIndxLeft].BoundingBox.zmax;
			right = parent->childTypeRight == INTERNALNODE ? internalNodes[parent->childIndxRight].BoundingBox.zmax : leafNodes[parent->childIndxRight].BoundingBox.zmax;
			buf.zmax = fmax(left, right);
			left = parent->childTypeLeft == INTERNALNODE ? internalNodes[parent->childIndxLeft].BoundingBox.zmin : leafNodes[parent->childIndxLeft].BoundingBox.zmin;
			right = parent->childTypeRight == INTERNALNODE ? internalNodes[parent->childIndxRight].BoundingBox.zmin : leafNodes[parent->childIndxRight].BoundingBox.zmin;
			buf.zmin = fmin(left, right);

			parent->BoundingBox = buf;
			ptr = parent;
			unsigned int oldpid = parentIdx;
			parentIdx = ptr->getParent();
			if (ptr->ParentIndex > -1)
			{
				parent = &internalNodes[parentIdx];
				//printf("In Internal Node %d\t, xmin: %f, parent: %d\n", oldpid, buf.xmin, parentIdx);

			}
			else
			{
				break;
			}
		}

	}

	struct CollisionPair
	{
		int idx1;
		int idx2;
	};

	struct CollisionList
	{
		//int* list;

		//__device__ void add(int x1, int x2)
		//{

		//}
		//thrust::dev<CollisionPair> list;
		//__device__ void Add(int idx1, int idx2)
		//{
		//	list.push_back({ idx1,idx2 });
		//}
	};

	__host__ __device__ bool AABBTest(BBox& a, BBox& b)
	{
		return !(a.xmin > b.xmax || b.xmin > a.xmax ||
			a.ymin > b.ymax || b.ymin > a.ymax ||
			a.zmin > b.zmax || b.zmin > a.zmax);
	}

#define ALLOC_OFFSET 80

	__device__ bool CullNeightbor(LeafNode* a, LeafNode* b)
	{
		return !((
			a->p1 == b->p1 || a->p1 == b->p2 || a->p1 == b->p3 ||
			a->p2 == b->p1 || a->p2 == b->p2 || a->p2 == b->p3 ||
			a->p3 == b->p1 || a->p3 == b->p2 || a->p3 == b->p3));
	}

	__device__ void traverseIterative(int* list, int offset_start,
		InternalNode* internalNodes, LeafNode* leafNodes,
		LeafNode& queryNode,
		int queryObjectIdx)
	{
		// Allocate traversal stack from thread-local memory,
		// and push NULL to indicate that there are no postponed nodes.
		InternalNode* stack[64];
		InternalNode** stackPtr = stack;
		*stackPtr++ = NULL; // push

		int offset = 0;
		// Traverse nodes starting from the root.
		InternalNode* node = &internalNodes[0];
		do
		{
			// Check each child node for overlap.
			Node* childL = node->childTypeLeft == INTERNALNODE ? (Node*)&internalNodes[node->childIndxLeft] : (Node*)&leafNodes[node->childIndxLeft];
			Node* childR = node->childTypeRight == INTERNALNODE ? (Node*)&internalNodes[node->childIndxRight] : (Node*)&leafNodes[node->childIndxRight];

			bool overlapL = (AABBTest(queryNode.BoundingBox, childL->BoundingBox));
			bool overlapR = (AABBTest(queryNode.BoundingBox, childR->BoundingBox));

			// Query overlaps a leaf node => report collision.
			if (overlapL && node->childTypeLeft == LEAFNODE && (queryNode.Index < childL->Index) && CullNeightbor(&queryNode, (LeafNode*)childL))
			{
				if (queryNode.Index == 0 && childL->Index == 3)
				{
					printf("%d, %d, %d, %d, %d, %d\d", queryNode.p1, queryNode.p2, queryNode.p3, ((LeafNode*)childL)->p1, ((LeafNode*)childL)->p2, ((LeafNode*)childL)->p3);
				}
				list[offset_start + offset++] = queryNode.Index;
				list[offset_start + offset++] = childL->Index;
			}

			if (overlapR && node->childTypeRight == LEAFNODE && (queryNode.Index < childL->Index) && CullNeightbor(&queryNode, (LeafNode*)childR))
			{
				list[offset_start + offset++] = queryNode.Index;
				list[offset_start + offset++] = childR->Index;
			}

			// Query overlaps an internal node => traverse.
			bool traverseL = (overlapL && node->childTypeLeft == INTERNALNODE);
			bool traverseR = (overlapR && node->childTypeRight == INTERNALNODE);

			if (!traverseL && !traverseR)
				node = *--stackPtr; // pop
			else
			{
				node = (traverseL) ? (InternalNode*)childL : (InternalNode*)childR;
				if (traverseL && traverseR)
					*stackPtr++ = (InternalNode*)childR; // push
			}
		} while (node != NULL);

		if (offset > ALLOC_OFFSET)
		{
			printf("bad alloc");
		}

	}

	__global__ void findPotentialCollisions(int* list,
		InternalNode* internalNodes, LeafNode* leafNodes,
		int queryObjectIdx)
	{
		int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx < queryObjectIdx)
		{
			LeafNode* leaf = &leafNodes[idx];
			traverseIterative(list, idx * ALLOC_OFFSET,
				internalNodes, leafNodes,
				*leaf,
				queryObjectIdx);
		}
	}

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

	__global__ void TrianglePairCollisionDetectionKernel(
		int* output, int* input, 
		LeafNode* leafNodes,
		int pairsize)
	{
		int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx < pairsize)
		{
			LeafNode* left = &leafNodes[input[idx * 2 + 0]];
			LeafNode* right = &leafNodes[input[idx * 2 + 1]];

			if (tri_contact(
				make_float3(left->v1_x, left->v1_y, left->v1_z),
				make_float3(left->v2_x, left->v2_y, left->v2_z),
				make_float3(left->v3_x, left->v3_y, left->v3_z),
				make_float3(right->v1_x, right->v1_y, right->v1_z),
				make_float3(right->v2_x, right->v2_y, right->v2_z),
				make_float3(right->v3_x, right->v3_y, right->v3_z)))
			{
				//if (left->primitiveIdx == 105439 && right->primitiveIdx == 1138346)
				//	printf("happy %d, %d", left->Index, right->Index);
				output[idx * 2 + 0] = left->primitiveIdx;
				output[idx * 2 + 1] = right->primitiveIdx;
			}
		}
	}

	//__global__ void prescan(float* g_odata, float* g_idata, int n) 
	//{
	//	extern __shared__ float temp[];  // allocated on invocation 
	//	int thid = threadIdx.x;;
	//	int offset = 1; 
	//	temp[2 * thid] = g_idata[2 * thid]; // load input into shared memory 
	//	temp[2 * thid + 1] = g_idata[2 * thid + 1];
	//	for (int d = n >> 1; d > 0; d >>= 1)                    // build sum in place up the tree 
	//	{ 
	//		__syncthreads();    
	//		if (thid < d)    { 
	//		int ai = offset * (2 * thid + 1) - 1;
	//		int bi = offset * (2 * thid + 2) - 1;
	//		temp[bi] += temp[ai];
	//	}
	//		
	//	offset *= 2; }

	//	if (thid == 0) { temp[n - 1] = 0; } // clear the last element  

	//	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan 
	//	{      
	//		offset >>= 1;      
	//		__syncthreads();      
	//		if (thid < d)      
	//		{ 
	//			int ai = offset * (2 * thid + 1) - 1;
	//			int bi = offset * (2 * thid + 2) - 1;
	//			float t = temp[ai]; 
	//			temp[ai] = temp[bi]; 
	//			temp[bi] += t;       
	//		} 
	//	}
	//	__syncthreads();
	//	g_odata[2 * thid] = temp[2 * thid]; // write results to device memory      
	//	g_odata[2*thid+1] = temp[2*thid+1]; 
	//}

	//__global__ void BVHCullingKernel(CollisionList* list,
	//	InternalNode* internalNodes, LeafNode* leafNodes,
	//	int triangleCount)
	//{
	//	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	//	if (idx < triangleCount)
	//	{
	//		list->Add(idx, idx);
	//		//traverseRecursive(bvh, list, objectAABBs[idx],
	//		//	idx, bvh.getRoot());

	//	}
	//}
#define BeginEvent(x)  
#define EndEvent(x, str)  
//#define BeginEvent(x)	cudaEvent_t start_##x, stop_##x; \
//						cudaEventCreate(&start_##x);\
//						cudaEventCreate(&stop_##x); \
//						cudaDeviceSynchronize(); \
//						cudaEventRecord(start_##x);
//#define EndEvent(x, str) cudaEventRecord(stop_##x);\
//						cudaEventSynchronize(stop_##x);\
//						float milliseconds_##x = 0;\
//						cudaEventElapsedTime(&milliseconds_##x, start_##x, stop_##x);\
//						cudaEventDestroy(start_##x);\
//						cudaEventDestroy(stop_##x);\
//						printf(str, milliseconds_##x);

	struct notminusone
	{
		__host__ __device__
			bool operator()(const int x)
		{
			return x != -1;
		}
	};

	void CuBVH::BuildBVH(const std::vector<Triangle>& triangles, const std::vector<float>& bbs)
	{
		clock_t start, end;
		start = std::clock();		//程序开始计时

		unsigned int triangleCount = triangles.size();

		Triangle* d_triangles;
		float* d_boundingboxes;
		MortonType* d_keys;
		MortonRec* d_records;

		checkCudaErrors(cudaMalloc((void**)&d_triangles, sizeof(Triangle) * triangles.size()));
		checkCudaErrors(cudaMalloc((void**)&d_boundingboxes, sizeof(float) * bbs.size()));
		checkCudaErrors(cudaMalloc((void**)&d_keys, sizeof(MortonType) * triangleCount));
		checkCudaErrors(cudaMalloc((void**)&d_records, sizeof(MortonRec) * triangleCount));

		checkCudaErrors(cudaMemcpy(d_triangles, triangles.data(), sizeof(Triangle) * triangles.size(), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_boundingboxes, bbs.data(), sizeof(float) * bbs.size(), cudaMemcpyHostToDevice));

		BeginEvent(Morton);
		Morton3DKernel<<<(triangleCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>
			(triangleCount, d_keys, d_boundingboxes, d_records);
		EndEvent(Morton, "It took me %f milliseconds to generate morton codes.\n");

		BeginEvent(Sorting);
		//wrap raw pointer with a device_ptr to use with Thrust functions
		thrust::device_ptr<MortonType> dev_keys_ptr(d_keys);
		thrust::device_ptr<MortonRec>  dev_data_ptr(d_records);
		thrust::sort_by_key(dev_keys_ptr, dev_keys_ptr + triangleCount, dev_data_ptr);
		d_keys = thrust::raw_pointer_cast(dev_keys_ptr);
		d_records = thrust::raw_pointer_cast(dev_data_ptr);
		EndEvent(Sorting, "It took me %f milliseconds to sort the morton codes.\n");

		BeginEvent(AssignLeaf);
		LeafNode* d_leafNodes;
		InternalNode* d_internalNodes;
		cudaMalloc((void**)&d_leafNodes, triangleCount * sizeof(LeafNode));
		cudaMalloc((void**)&d_internalNodes, (triangleCount - 1) * sizeof(InternalNode));
		AssignLeafNodesKernel << <(triangleCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >
			(triangleCount, d_leafNodes, d_records, d_boundingboxes, d_triangles);
		EndEvent(AssignLeaf, "It took me %f milliseconds to sort the assign leaf.\n");

		BeginEvent(AssignInternal);
		AssignInternalNodesKernel << <(triangleCount + THREADS_PER_BLOCK - 2) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >
			(triangleCount, d_keys, d_leafNodes, d_internalNodes, d_records);
		EndEvent(AssignInternal, "It took me %f milliseconds to sort the assign internal.\n");

		int* atom;
		cudaMalloc((void**)&atom, triangleCount * sizeof(int));
		cudaMemset(atom, 0, triangleCount * sizeof(int));

		BeginEvent(InternalNodeBBox);
		InternalNodeBBoxKernel << <(triangleCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >
			(triangleCount, atom, d_internalNodes, d_leafNodes);
		EndEvent(InternalNodeBBox, "It took me %f milliseconds to InternalNodeBBox.\n");

		LeafNode* h_leafNodes;
		InternalNode* h_internalNodes;
		h_leafNodes = (LeafNode*)malloc(triangleCount * sizeof(LeafNode));
		h_internalNodes = (InternalNode*)malloc((triangleCount - 1) * sizeof(InternalNode));
		checkCudaErrors(cudaMemcpy(h_leafNodes, d_leafNodes, triangleCount * sizeof(LeafNode), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_internalNodes, d_internalNodes, (triangleCount - 1) * sizeof(InternalNode), cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();

#pragma region CPU_SIMULTANEOUS_TRAVERSAL
		//CollisionList list;
		//BeginEvent(CullingBVH);
		//BVHCullingKernel << <(triangleCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >
		//	(&list, d_internalNodes, d_leafNodes, triangleCount);

		//EndEvent(CullingBVH, "It took me %f milliseconds to CullingBVH.\n");
		//std::vector<std::pair<int, int>> possible_pairs;
		//std::queue<std::pair<Node*, Node*>> test_pairs_queue;
		//test_pairs_queue.push({ &h_internalNodes[0], &h_internalNodes[0] });
		//while (test_pairs_queue.size() != 0)
		//{
		//	std::pair<Node*, Node*> pair = test_pairs_queue.front();
		//	test_pairs_queue.pop();

		//	if (pair.first == pair.second)
		//	{
		//		Node* lleft = h_internalNodes[pair.first->Index].childTypeLeft == INTERNALNODE ?
		//			(Node*)&h_internalNodes[h_internalNodes[pair.second->Index].childIndxLeft] :
		//			(Node*)&h_leafNodes[h_internalNodes[pair.second->Index].childIndxLeft];

		//		Node* lright = h_internalNodes[pair.first->Index].childTypeRight == INTERNALNODE ?
		//			(Node*)&h_internalNodes[h_internalNodes[pair.second->Index].childIndxRight] :
		//			(Node*)&h_leafNodes[h_internalNodes[pair.second->Index].childIndxRight];
		//		
		//		if(lleft->Type== INTERNALNODE)
		//			test_pairs_queue.push({ lleft, lleft });
		//		test_pairs_queue.push({ lleft, lright });
		//		if (lright->Type == INTERNALNODE)
		//			test_pairs_queue.push({ lright, lright });
		//	}
		//	else if (AABBTest(pair.first->BoundingBox, pair.second->BoundingBox))
		//	{
		//		if (pair.first->Type == LEAFNODE && pair.second->Type == LEAFNODE)
		//		{
		//			// ToDo: Cull adjacent triangle
		//			LeafNode* left = &h_leafNodes[pair.first->Index];
		//			LeafNode* right = &h_leafNodes[pair.second->Index];
		//			if (left->p1 == right->p1 || left->p1 == right->p2 || left->p1 == right->p3 || left->p2 == right->p2 || left->p2 == right->p3 || left->p3 == right->p3)
		//			{

		//			}
		//			else
		//			{
		//				if (left->primitiveIdx == 120914 || right->primitiveIdx == 120914)
		//				{
		//					std::cout << left->primitiveIdx << " " << right->primitiveIdx << std::endl;
		//				}
		//				possible_pairs.push_back({ pair.first->Index, pair.second->Index });
		//			}
		//		}
		//		else if (pair.first->Type == LEAFNODE)
		//		{
		//			Node* left = h_internalNodes[pair.second->Index].childTypeLeft == INTERNALNODE ? 
		//				(Node*)&h_internalNodes[h_internalNodes[pair.second->Index].childIndxLeft] : 
		//				(Node*)&h_leafNodes[h_internalNodes[pair.second->Index].childIndxLeft];

		//			Node* right = h_internalNodes[pair.second->Index].childTypeRight == INTERNALNODE ?
		//				(Node*)&h_internalNodes[h_internalNodes[pair.second->Index].childIndxRight] :
		//				(Node*)&h_leafNodes[h_internalNodes[pair.second->Index].childIndxRight];

		//			test_pairs_queue.push({ left, pair.first });
		//			test_pairs_queue.push({ right, pair.first });
		//		}
		//		else if (pair.second->Type == LEAFNODE)
		//		{
		//			Node* left = h_internalNodes[pair.first->Index].childTypeLeft == INTERNALNODE ?
		//				(Node*)&h_internalNodes[h_internalNodes[pair.first->Index].childIndxLeft] :
		//				(Node*)&h_leafNodes[h_internalNodes[pair.first->Index].childIndxLeft];

		//			Node* right = h_internalNodes[pair.first->Index].childTypeRight == INTERNALNODE ?
		//				(Node*)&h_internalNodes[h_internalNodes[pair.first->Index].childIndxRight] :
		//				(Node*)&h_leafNodes[h_internalNodes[pair.first->Index].childIndxRight];

		//			test_pairs_queue.push({ left, pair.second });
		//			test_pairs_queue.push({ right, pair.second });

		//		}
		//		else
		//		{
		//			// All internal node
		//			Node* lleft = h_internalNodes[pair.second->Index].childTypeLeft == INTERNALNODE ?
		//				(Node*)&h_internalNodes[h_internalNodes[pair.second->Index].childIndxLeft] :
		//				(Node*)&h_leafNodes[h_internalNodes[pair.second->Index].childIndxLeft];

		//			Node* lright = h_internalNodes[pair.second->Index].childTypeRight == INTERNALNODE ?
		//				(Node*)&h_internalNodes[h_internalNodes[pair.second->Index].childIndxRight] :
		//				(Node*)&h_leafNodes[h_internalNodes[pair.second->Index].childIndxRight];

		//			Node* rleft = h_internalNodes[pair.first->Index].childTypeLeft == INTERNALNODE ?
		//				(Node*)&h_internalNodes[h_internalNodes[pair.first->Index].childIndxLeft] :
		//				(Node*)&h_leafNodes[h_internalNodes[pair.first->Index].childIndxLeft];

		//			Node* rright = h_internalNodes[pair.first->Index].childTypeRight == INTERNALNODE ?
		//				(Node*)&h_internalNodes[h_internalNodes[pair.first->Index].childIndxRight] :
		//				(Node*)&h_leafNodes[h_internalNodes[pair.first->Index].childIndxRight];

		//			test_pairs_queue.push({ lleft, rleft });
		//			test_pairs_queue.push({ lleft, rright });
		//			test_pairs_queue.push({ lright, rleft });
		//			test_pairs_queue.push({ lright, rright });
		//		}
		//	}
		//	static bool first1 = true, first2 = true;
		//	if (first1 && test_pairs_queue.size() > 16384)
		//	{
		//		first1 = false;
		//		end = clock();		//程序结束用时
		//		double endtime = (double)(end - start) / CLOCKS_PER_SEC;
		//		std::cout << test_pairs_queue.size() << ": " << endtime << std::endl;
		//	}
		//	if (first2 &&  test_pairs_queue.size() > 131072)
		//	{
		//		first2 = false;
		//		end = clock();		//程序结束用时
		//		double endtime = (double)(end - start) / CLOCKS_PER_SEC;
		//		std::cout << test_pairs_queue.size() << ": " << endtime << std::endl;
		//	}
		//}
		//end = clock();		//程序结束用时
		//double endtime = (double)(end - start) / CLOCKS_PER_SEC;
		//std::cout << test_pairs_queue.size() << ": " << endtime << std::endl;
#pragma endregion

		BeginEvent(CullingBVH);
		int* d_possible_pair_list;
		cudaMalloc((void**)&d_possible_pair_list, triangleCount * ALLOC_OFFSET * sizeof(int));
		cudaMemset(d_possible_pair_list, -1, triangleCount * ALLOC_OFFSET * sizeof(int));
		findPotentialCollisions << <(triangleCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >
			(d_possible_pair_list, d_internalNodes, d_leafNodes, triangleCount);
		EndEvent(CullingBVH, "It took me %f milliseconds to CullingBVH.\n");

		BeginEvent(CullingMinusOne);
		int* d_filtered_pair_list;
		cudaMalloc((void**)&d_filtered_pair_list, triangleCount* ALLOC_OFFSET * sizeof(int));
		thrust::device_ptr<int> dev_possible_pair_ptr(d_possible_pair_list);
		thrust::device_ptr<int> dev_filtered_pair_ptr(d_filtered_pair_list);
		thrust::device_ptr<int> filter_res = thrust::copy_if(dev_possible_pair_ptr, dev_possible_pair_ptr + triangleCount * ALLOC_OFFSET, dev_filtered_pair_ptr, notminusone());
		unsigned int num_filter_pair_list = thrust::raw_pointer_cast(filter_res) - d_filtered_pair_list;
		num_filter_pair_list /= 2;
		EndEvent(CullingMinusOne, "It took me %f milliseconds to CullingMinusOne.\n");

		BeginEvent(FinalCheck);
		int* d_collided_pair_list;
		cudaMalloc((void**)&d_collided_pair_list, num_filter_pair_list* 2 * sizeof(int));
		cudaMemset(d_collided_pair_list, -1, num_filter_pair_list * 2 * sizeof(int));
		TrianglePairCollisionDetectionKernel << <(num_filter_pair_list + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >
		(
			d_collided_pair_list, d_filtered_pair_list,
			d_leafNodes,
			num_filter_pair_list);
		EndEvent(FinalCheck, "It took me %f milliseconds to Final Check.\n");

		BeginEvent(FinalFilter);
		int* d_filtered_collided_pairs;
		cudaMalloc((void**)&d_filtered_collided_pairs, num_filter_pair_list * 2 * sizeof(int));
		thrust::device_ptr<int> d_collided_pair_ptr(d_collided_pair_list);
		thrust::device_ptr<int> d_filtered_collided_ptr(d_filtered_collided_pairs);
		thrust::device_ptr<int> final_filter_res = thrust::copy_if(d_collided_pair_ptr, d_collided_pair_ptr + num_filter_pair_list * 2, d_filtered_collided_ptr, notminusone());
		unsigned int num_final_pair_nums = thrust::raw_pointer_cast(final_filter_res) - d_filtered_collided_pairs;
		num_final_pair_nums /= 2;
		EndEvent(FinalFilter, "It took me %f milliseconds to FinalFilter.\n");

		free(h_leafNodes);
		free(h_internalNodes);

		end = clock();		//程序结束用时
		double endtime = (double)(end - start) / CLOCKS_PER_SEC;
		std::cout << "Time period: " << endtime << std::endl;

		//Triangle t1 = triangles[d_filtered_collided_pairs[0]];
		//Triangle t2 = triangles[d_filtered_collided_pairs[1]];
	}
}