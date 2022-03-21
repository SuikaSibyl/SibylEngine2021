module;
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
export module Core.LinearAlgebra;
using namespace std;
#pragma warning(disable:4996)

//
// originally implemented by Justin Legakis
//

namespace SIByL
{
	inline namespace Core
	{
        class Matrix;

        // ====================================================================
        // ====================================================================

        export class Vec2f
        {
        public:

            // CONSTRUCTORS & DESTRUCTOR
            Vec2f() { data[0] = data[1] = 0; }
            Vec2f(const Vec2f& V) {
                data[0] = V.data[0];
                data[1] = V.data[1];
            }
            Vec2f(float d0, float d1) {
                data[0] = d0;
                data[1] = d1;
            }
            Vec2f(const Vec2f& V1, const Vec2f& V2) {
                data[0] = V1.data[0] - V2.data[0];
                data[1] = V1.data[1] - V2.data[1];
            }
            ~Vec2f() { }

            // ACCESSORS
            void Get(float& d0, float& d1) const {
                d0 = data[0];
                d1 = data[1];
            }
            float operator[](int i) const {
                assert(i >= 0 && i < 2);
                return data[i];
            }
            float x() const { return data[0]; }
            float y() const { return data[1]; }
            float Length() const {
                float l = (float)sqrt(data[0] * data[0] + data[1] * data[1]);
                return l;
            }

            // MODIFIERS
            void Set(float d0, float d1) {
                data[0] = d0;
                data[1] = d1;
            }
            void Scale(float d0, float d1) {
                data[0] *= d0;
                data[1] *= d1;
            }
            void Divide(float d0, float d1) {
                data[0] /= d0;
                data[1] /= d1;
            }
            void Negate() {
                data[0] = -data[0];
                data[1] = -data[1];
            }

            // OVERLOADED OPERATORS
            Vec2f& operator=(const Vec2f& V) {
                data[0] = V.data[0];
                data[1] = V.data[1];
                return *this;
            }
            int operator==(const Vec2f& V) const {
                return ((data[0] == V.data[0]) &&
                    (data[1] == V.data[1]));
            }
            int operator!=(const Vec2f& V) {
                return ((data[0] != V.data[0]) ||
                    (data[1] != V.data[1]));
            }
            Vec2f& operator+=(const Vec2f& V) {
                data[0] += V.data[0];
                data[1] += V.data[1];
                return *this;
            }
            Vec2f& operator-=(const Vec2f& V) {
                data[0] -= V.data[0];
                data[1] -= V.data[1];
                return *this;
            }
            Vec2f& operator*=(float f) {
                data[0] *= f;
                data[1] *= f;
                return *this;
            }
            Vec2f& operator/=(float f) {
                data[0] /= f;
                data[1] /= f;
                return *this;
            }

            // OPERATIONS
            float Dot2(const Vec2f& V) const {
                return data[0] * V.data[0] + data[1] * V.data[1];
            }

            // STATIC OPERATIONS
            static void Add(Vec2f& a, const Vec2f& b, const Vec2f& c) {
                a.data[0] = b.data[0] + c.data[0];
                a.data[1] = b.data[1] + c.data[1];
            }
            static void Sub(Vec2f& a, const Vec2f& b, const Vec2f& c) {
                a.data[0] = b.data[0] - c.data[0];
                a.data[1] = b.data[1] - c.data[1];
            }
            static void CopyScale(Vec2f& a, const Vec2f& b, float c) {
                a.data[0] = b.data[0] * c;
                a.data[1] = b.data[1] * c;
            }
            static void AddScale(Vec2f& a, const Vec2f& b, const Vec2f& c, float d) {
                a.data[0] = b.data[0] + c.data[0] * d;
                a.data[1] = b.data[1] + c.data[1] * d;
            }
            static void Average(Vec2f& a, const Vec2f& b, const Vec2f& c) {
                a.data[0] = (b.data[0] + c.data[0]) * 0.5f;
                a.data[1] = (b.data[1] + c.data[1]) * 0.5f;
            }
            static void WeightedSum(Vec2f& a, const Vec2f& b, float c, const Vec2f& d, float e) {
                a.data[0] = b.data[0] * c + d.data[0] * e;
                a.data[1] = b.data[1] * c + d.data[1] * e;
            }

            // INPUT / OUTPUT
            void Write(FILE* F = stdout) const {
                fprintf(F, "%f %f\n", data[0], data[1]);
            }

        private:
            // REPRESENTATION
            float		data[2];

        };

        // ====================================================================
        // ====================================================================

        export class Vec3f
        {
        public:

            // CONSTRUCTORS & DESTRUCTOR
            Vec3f() { data[0] = data[1] = data[2] = 0; }
            Vec3f(const Vec3f& V) {
                data[0] = V.data[0];
                data[1] = V.data[1];
                data[2] = V.data[2];
            }
            Vec3f(float d0, float d1, float d2) {
                data[0] = d0;
                data[1] = d1;
                data[2] = d2;
            }
            Vec3f(const Vec3f& V1, const Vec3f& V2) {
                data[0] = V1.data[0] - V2.data[0];
                data[1] = V1.data[1] - V2.data[1];
                data[2] = V1.data[2] - V2.data[2];
            }
            ~Vec3f() { }

            // ACCESSORS
            void Get(float& d0, float& d1, float& d2) const {
                d0 = data[0];
                d1 = data[1];
                d2 = data[2];
            }
            float operator[](int i) const {
                assert(i >= 0 && i < 3);
                return data[i];
            }
            float x() const { return data[0]; }
            float y() const { return data[1]; }
            float z() const { return data[2]; }
            float r() const { return data[0]; }
            float g() const { return data[1]; }
            float b() const { return data[2]; }
            float Length() const {
                float l = (float)sqrt(data[0] * data[0] +
                    data[1] * data[1] +
                    data[2] * data[2]);
                return l;
            }

            // MODIFIERS
            void Set(float d0, float d1, float d2) {
                data[0] = d0;
                data[1] = d1;
                data[2] = d2;
            }
            void Scale(float d0, float d1, float d2) {
                data[0] *= d0;
                data[1] *= d1;
                data[2] *= d2;
            }
            void Divide(float d0, float d1, float d2) {
                data[0] /= d0;
                data[1] /= d1;
                data[2] /= d2;
            }
            void Normalize() {
                float l = Length();
                if (l > 0) {
                    data[0] /= l;
                    data[1] /= l;
                    data[2] /= l;
                }
            }
            void Negate() {
                data[0] = -data[0];
                data[1] = -data[1];
                data[2] = -data[2];
            }
            void Clamp(float low = 0, float high = 1) {
                if (data[0] < low) data[0] = low;  if (data[0] > high) data[0] = high;
                if (data[1] < low) data[1] = low;  if (data[1] > high) data[1] = high;
                if (data[2] < low) data[2] = low;  if (data[2] > high) data[2] = high;
            }

            // OVERLOADED OPERATORS
            Vec3f& operator=(const Vec3f& V) {
                data[0] = V.data[0];
                data[1] = V.data[1];
                data[2] = V.data[2];
                return *this;
            }
            int operator==(const Vec3f& V) {
                return ((data[0] == V.data[0]) &&
                    (data[1] == V.data[1]) &&
                    (data[2] == V.data[2]));
            }
            int operator!=(const Vec3f& V) {
                return ((data[0] != V.data[0]) ||
                    (data[1] != V.data[1]) ||
                    (data[2] != V.data[2]));
            }
            Vec3f& operator+=(const Vec3f& V) {
                data[0] += V.data[0];
                data[1] += V.data[1];
                data[2] += V.data[2];
                return *this;
            }
            Vec3f& operator-=(const Vec3f& V) {
                data[0] -= V.data[0];
                data[1] -= V.data[1];
                data[2] -= V.data[2];
                return *this;
            }
            Vec3f& operator*=(int i) {
                data[0] = float(data[0] * i);
                data[1] = float(data[1] * i);
                data[2] = float(data[2] * i);
                return *this;
            }
            Vec3f& operator*=(float f) {
                data[0] *= f;
                data[1] *= f;
                data[2] *= f;
                return *this;
            }
            Vec3f& operator/=(int i) {
                data[0] = float(data[0] / i);
                data[1] = float(data[1] / i);
                data[2] = float(data[2] / i);
                return *this;
            }
            Vec3f& operator/=(float f) {
                data[0] /= f;
                data[1] /= f;
                data[2] /= f;
                return *this;
            }


            friend Vec3f operator+(const Vec3f& v1, const Vec3f& v2) {
                Vec3f v3; Add(v3, v1, v2); return v3;
            }
            friend Vec3f operator-(const Vec3f& v1, const Vec3f& v2) {
                Vec3f v3; Sub(v3, v1, v2); return v3;
            }
            friend Vec3f operator*(const Vec3f& v1, float f) {
                Vec3f v2; CopyScale(v2, v1, f); return v2;
            }
            friend Vec3f operator*(float f, const Vec3f& v1) {
                Vec3f v2; CopyScale(v2, v1, f); return v2;
            }
            friend Vec3f operator*(const Vec3f& v1, const Vec3f& v2) {
                Vec3f v3; Mult(v3, v1, v2); return v3;
            }


            // OPERATIONS
            float Dot3(const Vec3f& V) const {
                return data[0] * V.data[0] +
                    data[1] * V.data[1] +
                    data[2] * V.data[2];
            }

            // STATIC OPERATIONS
            static void Add(Vec3f& a, const Vec3f& b, const Vec3f& c) {
                a.data[0] = b.data[0] + c.data[0];
                a.data[1] = b.data[1] + c.data[1];
                a.data[2] = b.data[2] + c.data[2];
            }
            static void Sub(Vec3f& a, const Vec3f& b, const Vec3f& c) {
                a.data[0] = b.data[0] - c.data[0];
                a.data[1] = b.data[1] - c.data[1];
                a.data[2] = b.data[2] - c.data[2];
            }
            static void Mult(Vec3f& a, const Vec3f& b, const Vec3f& c) {
                a.data[0] = b.data[0] * c.data[0];
                a.data[1] = b.data[1] * c.data[1];
                a.data[2] = b.data[2] * c.data[2];
            }
            static void CopyScale(Vec3f& a, const Vec3f& b, float c) {
                a.data[0] = b.data[0] * c;
                a.data[1] = b.data[1] * c;
                a.data[2] = b.data[2] * c;
            }
            static void AddScale(Vec3f& a, const Vec3f& b, const Vec3f& c, float d) {
                a.data[0] = b.data[0] + c.data[0] * d;
                a.data[1] = b.data[1] + c.data[1] * d;
                a.data[2] = b.data[2] + c.data[2] * d;
            }
            static void Average(Vec3f& a, const Vec3f& b, const Vec3f& c) {
                a.data[0] = (b.data[0] + c.data[0]) * 0.5f;
                a.data[1] = (b.data[1] + c.data[1]) * 0.5f;
                a.data[2] = (b.data[2] + c.data[2]) * 0.5f;
            }
            static void WeightedSum(Vec3f& a, const Vec3f& b, float c, const Vec3f& d, float e) {
                a.data[0] = b.data[0] * c + d.data[0] * e;
                a.data[1] = b.data[1] * c + d.data[1] * e;
                a.data[2] = b.data[2] * c + d.data[2] * e;
            }
            static void Cross3(Vec3f& c, const Vec3f& v1, const Vec3f& v2) {
                float x = v1.data[1] * v2.data[2] - v1.data[2] * v2.data[1];
                float y = v1.data[2] * v2.data[0] - v1.data[0] * v2.data[2];
                float z = v1.data[0] * v2.data[1] - v1.data[1] * v2.data[0];
                c.data[0] = x; c.data[1] = y; c.data[2] = z;
            }

            static void Min(Vec3f& a, const Vec3f& b, const Vec3f& c) {
                a.data[0] = (b.data[0] < c.data[0]) ? b.data[0] : c.data[0];
                a.data[1] = (b.data[1] < c.data[1]) ? b.data[1] : c.data[1];
                a.data[2] = (b.data[2] < c.data[2]) ? b.data[2] : c.data[2];
            }
            static void Max(Vec3f& a, const Vec3f& b, const Vec3f& c) {
                a.data[0] = (b.data[0] > c.data[0]) ? b.data[0] : c.data[0];
                a.data[1] = (b.data[1] > c.data[1]) ? b.data[1] : c.data[1];
                a.data[2] = (b.data[2] > c.data[2]) ? b.data[2] : c.data[2];
            }

            // INPUT / OUTPUT
            void Write(FILE* F = stdout) const {
                fprintf(F, "%f %f %f\n", data[0], data[1], data[2]);
            }

        private:

            friend class Matrix;

            // REPRESENTATION
            float		data[3];

        };

        // ====================================================================
        // ====================================================================

        export class Vec4f
        {
        public:

            // CONSTRUCTORS & DESTRUCTOR
            Vec4f() { data[0] = data[1] = data[2] = data[3] = 0; }
            Vec4f(const Vec4f& V) {
                data[0] = V.data[0];
                data[1] = V.data[1];
                data[2] = V.data[2];
                data[3] = V.data[3];
            }
            Vec4f(float d0, float d1, float d2, float d3) {
                data[0] = d0;
                data[1] = d1;
                data[2] = d2;
                data[3] = d3;
            }
            Vec4f(const Vec3f& V, float w) {
                data[0] = V.x();
                data[1] = V.y();
                data[2] = V.z();
                data[3] = w;
            }
            Vec4f(const Vec4f& V1, const Vec4f& V2) {
                data[0] = V1.data[0] - V2.data[0];
                data[1] = V1.data[1] - V2.data[1];
                data[2] = V1.data[2] - V2.data[2];
                data[3] = V1.data[3] - V2.data[3];
            }
            ~Vec4f() { }

            // ACCESSORS
            void Get(float& d0, float& d1, float& d2, float& d3) const {
                d0 = data[0];
                d1 = data[1];
                d2 = data[2];
                d3 = data[3];
            }
            float operator[](int i) const {
                assert(i >= 0 && i < 4);
                return data[i];
            }
            float x() const { return data[0]; }
            float y() const { return data[1]; }
            float z() const { return data[2]; }
            float w() const { return data[3]; }
            float r() const { return data[0]; }
            float g() const { return data[1]; }
            float b() const { return data[2]; }
            float a() const { return data[3]; }
            float Length() const {
                float l = (float)sqrt(data[0] * data[0] +
                    data[1] * data[1] +
                    data[2] * data[2] +
                    data[3] * data[3]);
                return l;
            }

            // MODIFIERS
            void Set(float d0, float d1, float d2, float d3) {
                data[0] = d0;
                data[1] = d1;
                data[2] = d2;
                data[3] = d3;
            }
            void Scale(float d0, float d1, float d2, float d3) {
                data[0] *= d0;
                data[1] *= d1;
                data[2] *= d2;
                data[3] *= d3;
            }
            void Divide(float d0, float d1, float d2, float d3) {
                data[0] /= d0;
                data[1] /= d1;
                data[2] /= d2;
                data[3] /= d3;
            }
            void Negate() {
                data[0] = -data[0];
                data[1] = -data[1];
                data[2] = -data[2];
                data[3] = -data[3];
            }
            void Normalize() {
                float l = Length();
                if (l > 0) {
                    data[0] /= l;
                    data[1] /= l;
                    data[2] /= l;
                }
            }
            void DivideByW() {
                if (data[3] != 0) {
                    data[0] /= data[3];
                    data[1] /= data[3];
                    data[2] /= data[3];
                }
                else {
                    data[0] = data[1] = data[2] = 0;
                }
                data[3] = 1;
            }

            // OVERLOADED OPERATORS
            Vec4f& operator=(const Vec4f& V) {
                data[0] = V.data[0];
                data[1] = V.data[1];
                data[2] = V.data[2];
                data[3] = V.data[3];
                return *this;
            }
            int operator==(const Vec4f& V) const {
                return ((data[0] == V.data[0]) &&
                    (data[1] == V.data[1]) &&
                    (data[2] == V.data[2]) &&
                    (data[3] == V.data[3]));
            }
            int operator!=(const Vec4f& V) const {
                return ((data[0] != V.data[0]) ||
                    (data[1] != V.data[1]) ||
                    (data[2] != V.data[2]) ||
                    (data[3] != V.data[3]));
            }
            Vec4f& operator+=(const Vec4f& V) {
                data[0] += V.data[0];
                data[1] += V.data[1];
                data[2] += V.data[2];
                data[3] += V.data[3];
                return *this;
            }
            Vec4f& operator-=(const Vec4f& V) {
                data[0] -= V.data[0];
                data[1] -= V.data[1];
                data[2] -= V.data[2];
                data[3] -= V.data[3];
                return *this;
            }
            Vec4f& operator*=(float f) {
                data[0] *= f;
                data[1] *= f;
                data[2] *= f;
                data[3] *= f;
                return *this;
            }
            Vec4f& operator/=(float f) {
                data[0] /= f;
                data[1] /= f;
                data[2] /= f;
                data[3] /= f;
                return *this;
            }

            // OPERATIONS
            float Dot2(const Vec4f& V) const {
                return data[0] * V.data[0] +
                    data[1] * V.data[1];
            }
            float Dot3(const Vec4f& V) const {
                return data[0] * V.data[0] +
                    data[1] * V.data[1] +
                    data[2] * V.data[2];
            }
            float Dot4(const Vec4f& V) const {
                return data[0] * V.data[0] +
                    data[1] * V.data[1] +
                    data[2] * V.data[2] +
                    data[3] * V.data[3];
            }

            // STATIC OPERATIONS
            static void Add(Vec4f& a, const Vec4f& b, const Vec4f& c) {
                a.data[0] = b.data[0] + c.data[0];
                a.data[1] = b.data[1] + c.data[1];
                a.data[2] = b.data[2] + c.data[2];
                a.data[3] = b.data[3] + c.data[3];
            }
            static void Sub(Vec4f& a, const Vec4f& b, const Vec4f& c) {
                a.data[0] = b.data[0] - c.data[0];
                a.data[1] = b.data[1] - c.data[1];
                a.data[2] = b.data[2] - c.data[2];
                a.data[3] = b.data[3] - c.data[3];
            }
            static void CopyScale(Vec4f& a, const Vec4f& b, float c) {
                a.data[0] = b.data[0] * c;
                a.data[1] = b.data[1] * c;
                a.data[2] = b.data[2] * c;
                a.data[3] = b.data[3] * c;
            }
            static void AddScale(Vec4f& a, const Vec4f& b, const Vec4f& c, float d) {
                a.data[0] = b.data[0] + c.data[0] * d;
                a.data[1] = b.data[1] + c.data[1] * d;
                a.data[2] = b.data[2] + c.data[2] * d;
                a.data[3] = b.data[3] + c.data[3] * d;
            }
            static void Average(Vec4f& a, const Vec4f& b, const Vec4f& c) {
                a.data[0] = (b.data[0] + c.data[0]) * 0.5f;
                a.data[1] = (b.data[1] + c.data[1]) * 0.5f;
                a.data[2] = (b.data[2] + c.data[2]) * 0.5f;
                a.data[3] = (b.data[3] + c.data[3]) * 0.5f;
            }
            static void WeightedSum(Vec4f& a, const Vec4f& b, float c, const Vec4f& d, float e) {
                a.data[0] = b.data[0] * c + d.data[0] * e;
                a.data[1] = b.data[1] * c + d.data[1] * e;
                a.data[2] = b.data[2] * c + d.data[2] * e;
                a.data[3] = b.data[3] * c + d.data[3] * e;
            }
            static void Cross3(Vec4f& c, const Vec4f& v1, const Vec4f& v2) {
                float x = v1.data[1] * v2.data[2] - v1.data[2] * v2.data[1];
                float y = v1.data[2] * v2.data[0] - v1.data[0] * v2.data[2];
                float z = v1.data[0] * v2.data[1] - v1.data[1] * v2.data[0];
                c.data[0] = x; c.data[1] = y; c.data[2] = z;
            }

            // INPUT / OUTPUT
            void Write(FILE* F = stdout) const {
                fprintf(F, "%f %f %f %f\n", data[0], data[1], data[2], data[3]);
            }

        private:

            friend class Matrix;

            // REPRESENTATION
            float		data[4];

        };

        inline ostream& operator<<(ostream& os, const Vec3f& v) {
            os << "Vec3f <" << v.x() << ", " << v.y() << ", " << v.z() << ">";
            return os;
        }


        export class Matrix
        {
        public:

            // CONSTRUCTORS & DESTRUCTOR
            Matrix() { Clear(); }
            Matrix(const Matrix& m);
            Matrix(const float* m);
            ~Matrix() {}

            // ACCESSORS
            float* glGet(void) const {
                float* glMat = new float[16];
                glMat[0] = data[0][0];  glMat[1] = data[1][0];  glMat[2] = data[2][0];  glMat[3] = data[3][0];
                glMat[4] = data[0][1];  glMat[5] = data[1][1];  glMat[6] = data[2][1];  glMat[7] = data[3][1];
                glMat[8] = data[0][2];  glMat[9] = data[1][2]; glMat[10] = data[2][2]; glMat[11] = data[3][2];
                glMat[12] = data[0][3]; glMat[13] = data[1][3]; glMat[14] = data[2][3]; glMat[15] = data[3][3];
                return glMat;
            }
            float Get(int x, int y) const {
                assert(x >= 0 && x < 4);
                assert(y >= 0 && y < 4);
                return data[y][x];
            }

            // MODIFIERS
            void Set(int x, int y, float v) {
                assert(x >= 0 && x < 4);
                assert(y >= 0 && y < 4);
                data[y][x] = v;
            }
            void SetToIdentity();
            void Clear();

            void Transpose(Matrix& m) const;
            void Transpose() { Transpose(*this); }

            int Inverse(Matrix& m, float epsilon = 1e-08) const;
            int Inverse(float epsilon = 1e-08) { return Inverse(*this, epsilon); }

            // OVERLOADED OPERATORS
            Matrix& operator=(const Matrix& m);
            int operator==(const Matrix& m) const;
            int operator!=(const Matrix& m) const { return !(*this == m); }
            friend Matrix operator+(const Matrix& m1, const Matrix& m2);
            friend Matrix operator-(const Matrix& m1, const Matrix& m2);
            friend Matrix operator*(const Matrix& m1, const Matrix& m2);
            friend Matrix operator*(const Matrix& m1, float f);
            friend Matrix operator*(float f, const Matrix& m) { return m * f; }
            Matrix& operator+=(const Matrix& m) { *this = *this + m; return *this; }
            Matrix& operator-=(const Matrix& m) { *this = *this - m; return *this; }
            Matrix& operator*=(const float f) { *this = *this * f; return *this; }
            Matrix& operator*=(const Matrix& m) { *this = *this * m; return *this; }

            // TRANSFORMATIONS
            static Matrix MakeTranslation(const Vec3f& v);
            static Matrix MakeScale(const Vec3f& v);
            static Matrix MakeScale(float s) { return MakeScale(Vec3f(s, s, s)); }
            static Matrix MakeXRotation(float theta);
            static Matrix MakeYRotation(float theta);
            static Matrix MakeZRotation(float theta);
            static Matrix MakeAxisRotation(const Vec3f& v, float theta);

            // Use to transform a point with a matrix
            // that may include translation
            void Transform(Vec4f& v) const;
            void Transform(Vec3f& v) const {
                Vec4f v2 = Vec4f(v.x(), v.y(), v.z(), 1);
                Transform(v2);
                v.Set(v2.x(), v2.y(), v2.z());
            }
            void Transform(Vec2f& v) const {
                Vec4f v2 = Vec4f(v.x(), v.y(), 1, 1);
                Transform(v2);
                v.Set(v2.x(), v2.y());
            }

            // Use to transform the direction of the ray
            // (ignores any translation)
            void TransformDirection(Vec3f& v) const {
                Vec4f v2 = Vec4f(v.x(), v.y(), v.z(), 0);
                Transform(v2);
                v.Set(v2.x(), v2.y(), v2.z());
            }

            // INPUT / OUTPUT
            void Write(FILE* F = stdout) const;
            void Write3x3(FILE* F = stdout) const;
            void Read(FILE* F);
            void Read3x3(FILE* F);

        private:

            // REPRESENTATION
            float	data[4][4];

        };


        float det4x4(float a1, float a2, float a3, float a4,
            float b1, float b2, float b3, float b4,
            float c1, float c2, float c3, float c4,
            float d1, float d2, float d3, float d4);
        float det3x3(float a1, float a2, float a3,
            float b1, float b2, float b3,
            float c1, float c2, float c3);
        float det2x2(float a, float b,
            float c, float d);

        // ===================================================================
        // ===================================================================
        // COPY CONSTRUCTOR

        Matrix::Matrix(const Matrix& m) {
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    data[y][x] = m.data[y][x];
                }
            }
        }

        Matrix::Matrix(const float* m) {
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    data[y][x] = m[4 * y + x];
                }
            }
        }

        // ===================================================================
        // ===================================================================
        // MODIFIERS

        void Matrix::SetToIdentity() {
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    data[y][x] = (x == y);
                }
            }
        }

        void Matrix::Clear() {
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    data[y][x] = 0;
                }
            }
        }

        void Matrix::Transpose(Matrix& m) const {
            // be careful, <this> might be <m>
            Matrix tmp = Matrix(*this);
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    m.data[y][x] = tmp.data[x][y];
                }
            }
        }

        // ===================================================================
        // ===================================================================
        // INVERSE

        int Matrix::Inverse(Matrix& m, float epsilon) const {
            m = *this;

            float a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, d1, d2, d3, d4;
            a1 = m.data[0][0]; b1 = m.data[0][1]; c1 = m.data[0][2]; d1 = m.data[0][3];
            a2 = m.data[1][0]; b2 = m.data[1][1]; c2 = m.data[1][2]; d2 = m.data[1][3];
            a3 = m.data[2][0]; b3 = m.data[2][1]; c3 = m.data[2][2]; d3 = m.data[2][3];
            a4 = m.data[3][0]; b4 = m.data[3][1]; c4 = m.data[3][2]; d4 = m.data[3][3];

            float det = det4x4(a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, d1, d2, d3, d4);

            if (fabs(det) < epsilon) {
                printf("Matrix::Inverse --- singular matrix, can't invert!\n");
                assert(0);
                return 0;
            }

            m.data[0][0] = det3x3(b2, b3, b4, c2, c3, c4, d2, d3, d4);
            m.data[1][0] = -det3x3(a2, a3, a4, c2, c3, c4, d2, d3, d4);
            m.data[2][0] = det3x3(a2, a3, a4, b2, b3, b4, d2, d3, d4);
            m.data[3][0] = -det3x3(a2, a3, a4, b2, b3, b4, c2, c3, c4);

            m.data[0][1] = -det3x3(b1, b3, b4, c1, c3, c4, d1, d3, d4);
            m.data[1][1] = det3x3(a1, a3, a4, c1, c3, c4, d1, d3, d4);
            m.data[2][1] = -det3x3(a1, a3, a4, b1, b3, b4, d1, d3, d4);
            m.data[3][1] = det3x3(a1, a3, a4, b1, b3, b4, c1, c3, c4);

            m.data[0][2] = det3x3(b1, b2, b4, c1, c2, c4, d1, d2, d4);
            m.data[1][2] = -det3x3(a1, a2, a4, c1, c2, c4, d1, d2, d4);
            m.data[2][2] = det3x3(a1, a2, a4, b1, b2, b4, d1, d2, d4);
            m.data[3][2] = -det3x3(a1, a2, a4, b1, b2, b4, c1, c2, c4);

            m.data[0][3] = -det3x3(b1, b2, b3, c1, c2, c3, d1, d2, d3);
            m.data[1][3] = det3x3(a1, a2, a3, c1, c2, c3, d1, d2, d3);
            m.data[2][3] = -det3x3(a1, a2, a3, b1, b2, b3, d1, d2, d3);
            m.data[3][3] = det3x3(a1, a2, a3, b1, b2, b3, c1, c2, c3);

            m *= 1 / det;
            return 1;
        }

        float det4x4(float a1, float a2, float a3, float a4,
            float b1, float b2, float b3, float b4,
            float c1, float c2, float c3, float c4,
            float d1, float d2, float d3, float d4) {
            return
                a1 * det3x3(b2, b3, b4, c2, c3, c4, d2, d3, d4)
                - b1 * det3x3(a2, a3, a4, c2, c3, c4, d2, d3, d4)
                + c1 * det3x3(a2, a3, a4, b2, b3, b4, d2, d3, d4)
                - d1 * det3x3(a2, a3, a4, b2, b3, b4, c2, c3, c4);
        }

        float det3x3(float a1, float a2, float a3,
            float b1, float b2, float b3,
            float c1, float c2, float c3) {
            return
                a1 * det2x2(b2, b3, c2, c3)
                - b1 * det2x2(a2, a3, c2, c3)
                + c1 * det2x2(a2, a3, b2, b3);
        }

        float det2x2(float a, float b,
            float c, float d) {
            return a * d - b * c;
        }

        // ===================================================================
        // ===================================================================
        // OVERLOADED OPERATORS

        Matrix& Matrix::operator=(const Matrix& m) {
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    data[y][x] = m.data[y][x];
                }
            }
            return (*this);
        }

        int Matrix::operator==(const Matrix& m) const {
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    if (this->data[y][x] != m.data[y][x]) {
                        return 0;
                    }
                }
            }
            return 1;
        }

        Matrix operator+(const Matrix& m1, const Matrix& m2) {
            Matrix answer;
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    answer.data[y][x] = m1.data[y][x] + m2.data[y][x];
                }
            }
            return answer;
        }

        Matrix operator-(const Matrix& m1, const Matrix& m2) {
            Matrix answer;
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    answer.data[y][x] = m1.data[y][x] - m2.data[y][x];
                }
            }
            return answer;
        }

        Matrix operator*(const Matrix& m1, const Matrix& m2) {
            Matrix answer;
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    for (int i = 0; i < 4; i++) {
                        answer.data[y][x]
                            += m1.data[y][i] * m2.data[i][x];
                    }
                }
            }
            return answer;
        }

        Matrix operator*(const Matrix& m, float f) {
            Matrix answer;
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    answer.data[y][x] = m.data[y][x] * f;
                }
            }
            return answer;
        }

        // ====================================================================
        // ====================================================================
        // TRANSFORMATIONS

        Matrix Matrix::MakeTranslation(const Vec3f& v) {
            Matrix t;
            t.SetToIdentity();
            t.data[0][3] = v.x();
            t.data[1][3] = v.y();
            t.data[2][3] = v.z();
            return t;
        }

        Matrix Matrix::MakeScale(const Vec3f& v) {
            Matrix s;
            s.SetToIdentity();
            s.data[0][0] = v.x();
            s.data[1][1] = v.y();;
            s.data[2][2] = v.z();
            s.data[3][3] = 1;
            return s;
        }

        Matrix Matrix::MakeXRotation(float theta) {
            Matrix rx;
            rx.SetToIdentity();
            rx.data[1][1] = (float)cos((float)theta);
            rx.data[1][2] = -(float)sin((float)theta);
            rx.data[2][1] = (float)sin((float)theta);
            rx.data[2][2] = (float)cos((float)theta);
            return rx;
        }

        Matrix Matrix::MakeYRotation(float theta) {
            Matrix ry;
            ry.SetToIdentity();
            ry.data[0][0] = (float)cos((float)theta);
            ry.data[0][2] = (float)sin((float)theta);
            ry.data[2][0] = -(float)sin((float)theta);
            ry.data[2][2] = (float)cos((float)theta);
            return ry;
        }

        Matrix Matrix::MakeZRotation(float theta) {
            Matrix rz;
            rz.SetToIdentity();
            rz.data[0][0] = (float)cos((float)theta);
            rz.data[0][1] = -(float)sin((float)theta);
            rz.data[1][0] = (float)sin((float)theta);
            rz.data[1][1] = (float)cos((float)theta);
            return rz;
        }

        Matrix Matrix::MakeAxisRotation(const Vec3f& v, float theta) {
            Matrix r;
            r.SetToIdentity();

            float x = v.x(); float y = v.y(); float z = v.z();

            float c = cosf(theta);
            float s = sinf(theta);
            float xx = x * x;
            float xy = x * y;
            float xz = x * z;
            float yy = y * y;
            float yz = y * z;
            float zz = z * z;

            r.Set(0, 0, (1 - c) * xx + c);
            r.Set(0, 1, (1 - c) * xy + z * s);
            r.Set(0, 2, (1 - c) * xz - y * s);
            r.Set(0, 3, 0);

            r.Set(1, 0, (1 - c) * xy - z * s);
            r.Set(1, 1, (1 - c) * yy + c);
            r.Set(1, 2, (1 - c) * yz + x * s);
            r.Set(1, 3, 0);

            r.Set(2, 0, (1 - c) * xz + y * s);
            r.Set(2, 1, (1 - c) * yz - x * s);
            r.Set(2, 2, (1 - c) * zz + c);
            r.Set(2, 3, 0);

            r.Set(3, 0, 0);
            r.Set(3, 1, 0);
            r.Set(3, 2, 0);
            r.Set(3, 3, 1);

            return r;
        }

        // ====================================================================
        // ====================================================================

        void Matrix::Transform(Vec4f& v) const {
            Vec4f answer;
            for (int y = 0; y < 4; y++) {
                answer.data[y] = 0;
                for (int i = 0; i < 4; i++) {
                    answer.data[y] += data[y][i] * v[i];
                }
            }
            v = answer;
        }

        // ====================================================================
        // ====================================================================

        void Matrix::Write(FILE* F) const {
            assert(F != NULL);
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    float tmp = data[y][x];
                    if (fabs(tmp) < 0.00001) tmp = 0;
                    fprintf(F, "%12.6f ", tmp);
                }
                fprintf(F, "\n");
            }
        }

        void Matrix::Write3x3(FILE* F) const {
            assert(F != NULL);
            for (int y = 0; y < 4; y++) {
                if (y == 2) continue;
                for (int x = 0; x < 4; x++) {
                    if (x == 2) continue;
                    float tmp = data[y][x];
                    if (fabs(tmp) < 0.00001) tmp = 0;
                    fprintf(F, "%12.6f ", tmp);
                }
                fprintf(F, "\n");
            }
        }

        void Matrix::Read(FILE* F) {
            assert(F != NULL);
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    int scanned = fscanf(F, "%f", &data[y][x]);
                    assert(scanned == 1);
                }
            }
        }

        void Matrix::Read3x3(FILE* F) {
            assert(F != NULL);
            Clear();
            for (int y = 0; y < 4; y++) {
                if (y == 2) continue;
                for (int x = 0; x < 4; x++) {
                    if (x == 2) continue;
                    int scanned = fscanf(F, "%f", &data[y][x]);
                    assert(scanned == 1);
                }
            }
        }

        // ====================================================================
        // ====================================================================

        // ====================================================================
        // ====================================================================
	}
}