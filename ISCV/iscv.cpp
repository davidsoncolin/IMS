#define BOOST_PYTHON_STATIC_LIB
#pragma warning(push)
#pragma warning(disable : 4267) 
#pragma warning(disable : 4244) // Conversion from 'const double' to 'float' possible loss of data
#pragma warning(disable : 4018) // signed/unsigned missmatch
#pragma warning(disable : 4996) // Function call with parameters that may be unsafe - this call relies on the caller to check that the passed values are correct.

// Don't do #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION as it currently breaks ISCV

// Some computer vision functions that are too slow to code in python && the opencv seems to suck for
// Must compile on all platforms


// We need python wrappers, so use boost
#include <boost/python/def.hpp>
#include <boost/python/module.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/bind.hpp>
#include <numpy/arrayobject.h>
#include <assert.h>
#include <vector>
#include <algorithm> // max_element

using namespace boost::python;
using boost::int32_t;
using boost::uint16_t;
using boost::int64_t;

template<class I>
struct argsort_compare
{
	I _it;
	argsort_compare(I it) : _it(it) {}
	bool operator()(const int32_t a, const int32_t b) const { return _it[a] < _it[b]; }
};

template<class I>
void argsort(I begin, I end, std::vector<int32_t> &order)
{
	argsort_compare<I> comp(begin);
	order.resize(std::distance(begin,end));
	for (int i = 0; i < order.size(); ++i) order[i] = i;
	std::sort(order.begin(), order.end(), comp);
}

//#include <numpy/ndarrayobject.h>

// General functions for accessing nicely behaved numpy data in C++
// From here http://docs.scipy.org/doc/numpy/reference/c-api.dtype.html
template <typename T> struct NpType {};
template <> struct NpType<double>                {enum { type_code = NPY_DOUBLE };};
template <> struct NpType<unsigned char>         {enum { type_code = NPY_UINT8 };};
template <> struct NpType<int32_t>               {enum { type_code = NPY_INT32 };};
template <> struct NpType<uint16_t>              {enum { type_code = NPY_UINT16 };};
template <> struct NpType<float>                 {enum { type_code = NPY_FLOAT };};
template <> struct NpType<std::complex<double> > {enum { type_code = NPY_CDOUBLE };};

template<class T>
class Np1D {
	const PyArrayObject *_object;
public:
	size_t _size;
	T* _data;
	Np1D(object &y, int sz=-1)
	{
		_object = (PyArrayObject*)PyArray_FROM_O(y.ptr());
		if (!_object) throw std::runtime_error("Np1D: Not a numpy array");
		if (_object->nd != 1) throw std::runtime_error("Np1D: Must be 1D numpy array");
		if (_object->descr->elsize != sizeof(T)) throw std::runtime_error("Np1D: Wrong numpy dtype");
		if (!PyArray_ISCARRAY(_object)) throw std::runtime_error("Np1D: Numpy array is not contiguous");
		if (int(PyArray_TYPE(_object)) != int(NpType<T>::type_code)) {
			//std::cerr << int(PyArray_TYPE(_object)) << "  " << int(NpType<T>::type_code) <<  std::endl;
			if (int(PyArray_TYPE(_object)) == NPY_INT && int(NpType<T>::type_code) == NPY_INT32) {
				//std::cerr << "(W) Warning Np1D: numpy dtype is int, expected int32?" << std::endl;
			}
			else {
				throw std::runtime_error("Np1D: Wrong numpy dtype");
			}
		}
		_size = _object->dimensions[0];
		if (sz != -1 && _size != sz) throw std::runtime_error("Np1D: Wrong size");
		_data = reinterpret_cast<T*>(_object->data);
	}
	virtual ~Np1D() { Py_DECREF(_object); }
	T& operator[](size_t i) { assert(i >= 0 && i < _size); return _data[i]; }
	const T operator[](size_t i) const { assert(i >= 0 && i < _size); return _data[i]; }
	T max(const T &d) const {
		size_t size = _size;
		if (size < 1) return d;
		return *std::max_element(_data,_data+size);
	}
	bool any() { return _size > 0; }
};

template<class T>
class Np2D {
	const PyArrayObject *_object;
public:
	size_t _rows, _cols;
	T* _data;
	Np2D(object &y, int rows=-1, int cols=-1)
	{
		_object = (PyArrayObject*)PyArray_FROM_O(y.ptr());
		if (!_object) throw std::runtime_error("Np2D: Not a numpy array");
		if (_object->nd != 2) throw std::runtime_error("Np2D: Must be 2D numpy array");
		if (_object->descr->elsize != sizeof(T)) throw std::runtime_error("Np2D: Wrong numpy dtype");
		if (!PyArray_ISCARRAY(_object)) throw std::runtime_error("Np2D: Numpy array is not contiguous");
		if (PyArray_TYPE(_object) != NpType<T>::type_code) throw std::runtime_error("Np2D: Wrong numpy dtype");
		_rows = _object->dimensions[0];
		_cols = _object->dimensions[1];
		if (rows != -1 && _rows != rows) throw std::runtime_error("Np2D: Wrong number of rows");
		if (cols != -1 && _cols != cols) throw std::runtime_error("Np2D: Wrong number of cols");
		_data = reinterpret_cast<T*>(_object->data);
	}
	virtual ~Np2D() { Py_DECREF(_object); }
	T* operator[](size_t i) { assert(i >= 0 && i < _rows); return _data + i*_cols; }
	const T* operator[](size_t i) const { assert(i >= 0 && i < _rows); return _data + i*_cols; }
	T max(const T &d) const {
		size_t size = _rows*_cols;
		if (size < 1) return d;
		return *std::max_element(_data,_data+size);
	}
	bool any() { return (_rows != -1 && _cols != -1); }
};

template<class T>
class Np3D {
	const PyArrayObject *_object;
public:
	size_t _rows, _cols, _chans;
	T* _data;
	Np3D(object &y, int rows=-1, int cols=-1, int chans=-1)
	{
		_object = (PyArrayObject*)PyArray_FROM_O(y.ptr());
		if (!_object) throw std::runtime_error("Np3D: Not a numpy array");
		if (_object->nd != 3) throw std::runtime_error("Np3D: Must be 3D numpy array");
		if (_object->descr->elsize != sizeof(T)) throw std::runtime_error("Np3D: Wrong numpy dtype");
		if (!PyArray_ISCARRAY(_object)) throw std::runtime_error("Np3D: Numpy array is not contiguous");
		if (PyArray_TYPE(_object) != NpType<T>::type_code) throw std::runtime_error("Np3D: Wrong numpy dtype");
		_rows = _object->dimensions[0];
		_cols = _object->dimensions[1];
		_chans = _object->dimensions[2];
		if (rows != -1 && _rows != rows) throw std::runtime_error("Np3D: Wrong number of rows");
		if (cols != -1 && _cols != cols) throw std::runtime_error("Np3D: Wrong number of cols");
		if (chans != -1 && _chans != chans) throw std::runtime_error("Np3D: Wrong number of chans");
		_data = reinterpret_cast<T*>(_object->data);
	}
	virtual ~Np3D() { Py_DECREF(_object); }
	T* operator[](size_t i) { assert(i >= 0 && i < _rows); return _data + i*_cols*_chans; }
	const T* operator[](size_t i) const { assert(i >= 0 && i < _rows); return _data + i*_cols*_chans; }
	bool any() { return (_rows != -1 && _cols != -1 && _chans != -1); }
};

template<class T>
class Np4D {
	const PyArrayObject *_object;
public:
	size_t _items, _rows, _cols, _chans;
	T* _data;
	Np4D(object &y, int items=-1,int rows=-1, int cols=-1, int chans=-1)
	{
		_object = (PyArrayObject*)PyArray_FROM_O(y.ptr());
		if (!_object) throw std::runtime_error("Np4D: Not a numpy array");
		if (_object->nd != 4) throw std::runtime_error("Np4D: Must be 4D numpy array");
		if (_object->descr->elsize != sizeof(T)) throw std::runtime_error("Np4D: Wrong numpy dtype");
		if (!PyArray_ISCARRAY(_object)) throw std::runtime_error("Np4D: Numpy array is not contiguous");
		if (PyArray_TYPE(_object) != NpType<T>::type_code) throw std::runtime_error("Np4D: Wrong numpy dtype");
		_items = _object->dimensions[0];
		_rows = _object->dimensions[1];
		_cols = _object->dimensions[2];
		_chans = _object->dimensions[3];
		if (items != -1 && _items != items) throw std::runtime_error("Np4D: Wrong number of items");
		if (rows != -1 && _rows != rows) throw std::runtime_error("Np4D: Wrong number of rows");
		if (cols != -1 && _cols != cols) throw std::runtime_error("Np4D: Wrong number of cols");
		if (chans != -1 && _chans != chans) throw std::runtime_error("Np4D: Wrong number of chans");
		_data = reinterpret_cast<T*>(_object->data);
	}
	virtual ~Np4D() { Py_DECREF(_object); }
	T* operator[](size_t i) { assert(i >= 0 && i < _items); return _data + i*_rows*_cols*_chans; }
	const T* operator[](size_t i) const { assert(i >= 0 && i < _items); return _data + i*_rows*_cols*_chans; }
};


template<class T>
object newArray(const int size)
{
	npy_intp py_size = size;
	PyObject *pobj = PyArray_SimpleNew(1, &py_size, NpType<T>::type_code);
	return numeric::array( handle<>(pobj) );
}

template<class T>
object newArray2D(const int rows, const int cols)
{
	npy_intp py_size[2] = { rows, cols };
	PyObject *pobj = PyArray_SimpleNew(2, py_size, NpType<T>::type_code);
	return numeric::array( handle<>(pobj) );
}

template<class T>
object newArray3D(const int rows, const int cols, const int chans)
{
	npy_intp py_size[3] = { rows, cols, chans };
	PyObject *pobj = PyArray_SimpleNew(3, py_size, NpType<T>::type_code);
	return numeric::array( handle<>(pobj) );
}

template<class T>
object newArrayFromVector(const std::vector<T> &v)
{
	const int size = v.size();
	object y(newArray<T>(size));
	if (size) {
		Np1D<T> o(y);
		std::copy(v.begin(), v.end(), o._data);
	}
	return y;
}

template<class T>
object newArray2DFromVector(const std::vector<T> &v, const int stride)
{
	const int size = v.size() / stride;
	object y(newArray2D<T>(size, stride));
	if (size) {
		Np2D<T> o(y);
		std::copy(v.begin(), v.end(), o._data);
	}
	return y;
}


class ProjectVisibility
{
	Np2D<float> *_normals_x3d;
	Np3D<float> *_triangles;
	Np2D<float> *_Ts;
	Np2D<float> *_normals_tris;
	bool _useTriangles;

	float _intersectionThreshold;
	bool _generateNormals;

public:
	ProjectVisibility() : _intersectionThreshold(0.0), _generateNormals(false), _useTriangles(false), _normals_x3d(NULL), _triangles(NULL), _Ts(NULL), _normals_tris(NULL) {}
	~ProjectVisibility() { delete _normals_x3d, _triangles, _Ts, _normals_tris; }

	static boost::shared_ptr<ProjectVisibility> create()
	{
		return boost::shared_ptr<ProjectVisibility>(new ProjectVisibility());
	}

	void setNormals(object &o_normals_x3d)
	{
		delete _normals_x3d;
		_normals_x3d = new Np2D<float>(o_normals_x3d, -1, 3);
		_useTriangles = false;
	}

	void setLods(
		object &o_triangles, object &o_Ts, object &o_normals_tris,
		float intersectionThreshold = 100.0, bool generateNormals = true
		)
	{
		delete _triangles, _Ts, _normals_tris;
		_triangles = new Np3D<float>(o_triangles, -1, 3, 3);
		_useTriangles = true;
		_Ts = new Np2D<float>(o_Ts, -1, 3);
		_normals_tris = new Np2D<float>(o_normals_tris, -1, 3);
		_intersectionThreshold = intersectionThreshold;
		_generateNormals = generateNormals;
	}

	void setNormalsAndLods(
		object &o_normals_x3d, object &o_triangles, object &o_Ts, object &o_normals_tris,
		float intersectionThreshold = 100.0, bool generateNormals = true
		)
	{
		delete _normals_x3d, _triangles, _Ts, _normals_tris;
		_normals_x3d = new Np2D<float>(o_normals_x3d, -1, 3);
		_triangles = new Np3D<float>(o_triangles, -1, 3, 3);
		_useTriangles = true;
		_Ts = new Np2D<float>(o_Ts, -1, 3);
		_normals_tris = new Np2D<float>(o_normals_tris, -1, 3);
		_intersectionThreshold = intersectionThreshold;
		_generateNormals = generateNormals;
	}

	const float* normals()
	{
		if (_normals_x3d && _normals_x3d->any()) return _normals_x3d->_data;
		return NULL;
	}

	int numTriangles()
	{
		if (_useTriangles && _triangles->any()) return _triangles->_rows;
		return 0;
	}

	const float* triangles()
	{
		if (_triangles->any()) return _triangles->_data;
		return NULL;
	}

	const float* triangleNormals()
	{
		if (_normals_tris && _normals_tris->any()) return _normals_tris->_data;
		return NULL;
	}

	const float* T(int cameraIndex)
	{
		if (_Ts && _Ts->any()) return (*_Ts)[cameraIndex];
		return NULL;
	}

	float intersectionThreshold()
	{
		return _intersectionThreshold;
	}

	bool generateNormals()
	{
		return _generateNormals;
	}
};


namespace ISCV_python
{

// skeleton computation
// updates the chanMats and Gs
void pose_skeleton_with_chan_mats(object &o_chanMats, object &o_Gs, object &o_Ls, object &o_jointParents,
								  object &o_jointChans, object &o_jointChanSplits, object &o_chanValues,
								  object &o_root)
{
	const Np1D<float>   chanValues(o_chanValues, -1);
	const int     numChans = chanValues._size;
	Np3D<float>   chanMats(o_chanMats, numChans, 3, 4);
	Np3D<float>   Gs(o_Gs, -1, 3, 4);
	const int     numJoints = Gs._rows;
	const Np3D<float>   Ls(o_Ls, numJoints, 3, 4);
	const Np1D<int32_t> jointParents(o_jointParents, numJoints);
	const Np1D<int32_t> jointChanSplits(o_jointChanSplits, 2*numJoints+1);
	const Np1D<int32_t> jointChans(o_jointChans, jointChanSplits[2*numJoints]);
	const Np2D<float>   root(o_root, 3, 4);
	int cij[4] = { 1,2,0,1 };
	/*Fill in the channelMats and Gs from the dofValues (mm and radians) and skeleton definition.*/
	for (int ji = 0; ji < numJoints; ++ji) {
		float *Gs_ji = Gs[ji];
		const float *Ls_ji = Ls[ji];
		int pi = jointParents[ji];
		int nt = jointChanSplits[2*ji], nr = jointChanSplits[2*ji+1], ne = jointChanSplits[2*ji+2];
		const float *Gs_pi = (pi == -1? root._data:Gs[pi]);
		memcpy(Gs_ji, Gs_pi, 12*sizeof(float)); Gs_pi = Gs_ji;
		for (; nt < nr; ++nt) {
			int c = jointChans[nt];
			float *Ds_i = chanMats[nt];
			float v = chanValues[nt];
			for (int r = 0; r < 12; r+=4) Gs_ji[r+3] += Gs_ji[r+c] * v;
			memcpy(Ds_i, Gs_ji, 12*sizeof(float));
		}
		for (int r = 0; r < 12; r+=4) {
			const float Gs_pi_0 = Gs_pi[r+0], Gs_pi_1 = Gs_pi[r+1], Gs_pi_2 = Gs_pi[r+2];
			Gs_ji[r+0] = Gs_pi_0 * Ls_ji[4*0+0] + Gs_pi_1 * Ls_ji[4*1+0] + Gs_pi_2 * Ls_ji[4*2+0];
			Gs_ji[r+1] = Gs_pi_0 * Ls_ji[4*0+1] + Gs_pi_1 * Ls_ji[4*1+1] + Gs_pi_2 * Ls_ji[4*2+1];
			Gs_ji[r+2] = Gs_pi_0 * Ls_ji[4*0+2] + Gs_pi_1 * Ls_ji[4*1+2] + Gs_pi_2 * Ls_ji[4*2+2];
			Gs_ji[r+3] = Gs_pi_0 * Ls_ji[4*0+3] + Gs_pi_1 * Ls_ji[4*1+3] + Gs_pi_2 * Ls_ji[4*2+3] + Gs_pi[r+3];
		}
		// TODO, fix the order of the dofs so that we don't have to reverse the rotations here!
		for (nr = ne-1; nr >= nt; --nr) {
			int c = jointChans[nr]; // {3:rx,4:ry,5:rz}
			float *Ds_i = chanMats[nr];
			float v = chanValues[nr];
			float sv = sin(v), cv = cos(v); // TODO: sincos ?
			int ci = cij[c-3], cj = cij[c-2];
			float *Ri = Gs_ji+ci, *Rj = Gs_ji+cj;
			for (int r = 0; r < 12; r+=4) {
				float Ri_r_ = Ri[r];
				Ri[r] = Ri_r_*cv+Rj[r]*sv;
				Rj[r] = Rj[r]*cv-Ri_r_*sv;
			}
			memcpy(Ds_i, Gs_ji, 12*sizeof(float));
		}
	}
}

void pose_skeleton(object &o_Gs, object &o_Ls, object &o_jointParents,
				   object &o_jointChans, object &o_jointChanSplits, object &o_chanValues,
				   object &o_root)
{
	Np3D<float>   Gs(o_Gs, -1, 3, 4);
	const int     numJoints = Gs._rows;
	const Np3D<float>   Ls(o_Ls, numJoints, 3, 4);
	const Np1D<int32_t> jointParents(o_jointParents, numJoints);
	const Np1D<int32_t> jointChanSplits(o_jointChanSplits, 2*numJoints+1);
	const Np1D<int32_t> jointChans(o_jointChans, jointChanSplits[2*numJoints]);
	const Np1D<float>   chanValues(o_chanValues, -1);
	const int     numChans = chanValues._size;
	const Np2D<float>   root(o_root, 3, 4);
	const int cij[4] = { 1,2,0,1 };
	/*Fill in the channelMats and Gs from the dofValues (mm and radians) and skeleton definition.*/
	for (int ji = 0; ji < numJoints; ++ji) {
		float *Gs_ji = Gs[ji];
		const float *Ls_ji = Ls[ji];
		int pi = jointParents[ji];
		int nt = jointChanSplits[2*ji], nr = jointChanSplits[2*ji+1], ne = jointChanSplits[2*ji+2];
		const float *Gs_pi = (pi == -1? root._data:Gs[pi]);
		memcpy(Gs_ji, Gs_pi, 12*sizeof(float)); Gs_pi = Gs_ji;
		for (; nt < nr; ++nt) {
			int c = jointChans[nt];
			float v = chanValues[nt];
			for (int r = 0; r < 12; r+=4) Gs_ji[r+3] += Gs_ji[r+c] * v;
		}
		for (int r = 0; r < 12; r+=4) {
			const float Gs_pi_0 = Gs_pi[r+0], Gs_pi_1 = Gs_pi[r+1], Gs_pi_2 = Gs_pi[r+2];
			Gs_ji[r+0] = Gs_pi_0 * Ls_ji[4*0+0] + Gs_pi_1 * Ls_ji[4*1+0] + Gs_pi_2 * Ls_ji[4*2+0];
			Gs_ji[r+1] = Gs_pi_0 * Ls_ji[4*0+1] + Gs_pi_1 * Ls_ji[4*1+1] + Gs_pi_2 * Ls_ji[4*2+1];
			Gs_ji[r+2] = Gs_pi_0 * Ls_ji[4*0+2] + Gs_pi_1 * Ls_ji[4*1+2] + Gs_pi_2 * Ls_ji[4*2+2];
			Gs_ji[r+3] = Gs_pi_0 * Ls_ji[4*0+3] + Gs_pi_1 * Ls_ji[4*1+3] + Gs_pi_2 * Ls_ji[4*2+3] + Gs_pi[r+3];
		}
		// TODO, fix the order of the dofs so that we don't have to reverse the rotations here!
		for (nr = ne-1; nr >= nt; --nr) {
			int c = jointChans[nr]; // {3:rx,4:ry,5:rz}
			float v = chanValues[nr];
			float sv = sin(v), cv = cos(v);
			int ci = cij[c-3], cj = cij[c-2];
			float *Ri = Gs_ji+ci, *Rj = Gs_ji+cj;
			for (int r = 0; r < 12; r+=4) {
				float Ri_r_ = Ri[r];
				Ri[r] = Ri_r_*cv+Rj[r]*sv;
				Rj[r] = Rj[r]*cv-Ri_r_*sv;
			}
		}
	}
}

void copy_joints(object &o_src_Ls, object &o_src_jointChans, object &o_src_jointChanSplits, object &o_src_chanValues,
				 object &o_tgt_Ls, object &o_tgt_jointChans, object &o_tgt_jointChanSplits, object &o_tgt_chanValues,
				 object &o_jointMapping, object &o_jointSwizzles, object &o_jointOffsets)
				 {
	Np3D<float>   src_Ls(o_src_Ls, -1, 3, 4);
	const int     src_numJoints = src_Ls._rows;
	Np1D<int32_t> src_jointChanSplits(o_src_jointChanSplits, 2*src_numJoints+1);
	Np1D<int32_t> src_jointChans(o_src_jointChans, src_jointChanSplits[2*src_numJoints]);
	Np1D<float>   src_chanValues(o_src_chanValues, -1);

	Np3D<float>   tgt_Ls(o_tgt_Ls, -1, 3, 4);
	const int     tgt_numJoints = tgt_Ls._rows;
	Np1D<int32_t> tgt_jointChanSplits(o_tgt_jointChanSplits, 2*tgt_numJoints+1);
	Np1D<int32_t> tgt_jointChans(o_tgt_jointChans, tgt_jointChanSplits[2*tgt_numJoints]);
	Np1D<float>   tgt_chanValues(o_tgt_chanValues, -1);

	Np2D<int32_t> jointMapping(o_jointMapping, -1, 4); // src, tgt, swiz, off
	Np3D<float>   jointSwizzles(o_jointSwizzles, -1, 3, 3);
	Np3D<float>   jointOffsets(o_jointOffsets, -1, 3, 4);

	float M[12];
	for (int ji = 0; ji < jointMapping._rows; ++ji) {
		const int si = jointMapping[ji][0];
		const int ti = jointMapping[ji][1];
		const int swi = jointMapping[ji][2];
		const int ofi = jointMapping[ji][3];
		memcpy(M, src_Ls[si], 12*sizeof(float));
		int tran_from = src_jointChanSplits[2*si];
		int rot_from = src_jointChanSplits[2*si+1];
		int rot_end = src_jointChanSplits[2*si+2];
		for (int t = tran_from; t < rot_from; ++t) M[4*src_jointChans[t]+3] += src_chanValues[t];
		// TODO fixme, make the order of rotations be logical
		for (int ri = rot_end-1; ri >= rot_from; --ri) {
			const int ci = src_jointChans[ri];
			const float v = src_chanValues[ri];
			const float sv = sin(v), cv = cos(v); // sincos
			const int cj = (ci+1)%3, ck = (ci+2)%3;
			for (int i = 0; i < 12; i+= 4) {
				const float Rj = M[i+cj], Rk = M[i+ck];
				M[i+cj] = Rj*cv+Rk*sv;
				M[i+ck] = Rk*cv-Rj*sv;
			}
		}
		if (swi >= 0) {
			const float *swizzle = jointSwizzles[swi];
			// M_R = M_R * swizzle
			for (int r = 0; r < 12; r+=4) {
				const float M_0 = M[r+0], M_1 = M[r+1], M_2 = M[r+2];
				M[r+0] = M_0 * swizzle[0] + M_1 * swizzle[3] + M_2 * swizzle[6];
				M[r+1] = M_0 * swizzle[1] + M_1 * swizzle[4] + M_2 * swizzle[7];
				M[r+2] = M_0 * swizzle[2] + M_1 * swizzle[5] + M_2 * swizzle[8];
			}
			// M_R = swizzle^T M_R
			for (int i = 0; i < 3; i++) {
				const float M_0 = M[i], M_1 = M[4+i], M_2 = M[8+i];
				M[i  ] = swizzle[0] * M_0 + swizzle[3] * M_1 + swizzle[6] * M_2;
				M[4+i] = swizzle[1] * M_0 + swizzle[4] * M_1 + swizzle[7] * M_2;
				M[8+i] = swizzle[2] * M_0 + swizzle[5] * M_1 + swizzle[8] * M_2;
			}
			//M[:,:3] = np.dot(np.dot(swizzle.T, M[:,:3]),swizzle);
		}
		if (ofi >= 0) {
			const float *offset = jointOffsets[ofi];
			for (int r = 0; r < 12; r+=4) {
				const float M_0 = M[r+0], M_1 = M[r+1], M_2 = M[r+2];
				M[r+0] = M_0 * offset[4*0+0] + M_1 * offset[4*1+0] + M_2 * offset[4*2+0];
				M[r+1] = M_0 * offset[4*0+1] + M_1 * offset[4*1+1] + M_2 * offset[4*2+1];
				M[r+2] = M_0 * offset[4*0+2] + M_1 * offset[4*1+2] + M_2 * offset[4*2+2];
				M[r+3] = M_0 * offset[4*0+3] + M_1 * offset[4*1+3] + M_2 * offset[4*2+3] + M[r+3];
			}
		}
		float *L = tgt_Ls[ti];
		// M = T L R; M_T = T + L_T; M_R = L_R R_R
		for (int i = 0; i < 12; i += 4) M[i+3] -= L[i+3];

		// M_R = L_R^T M_R
		for (int i = 0; i < 3; i++) {
			const float M_0 = M[i], M_1 = M[4+i], M_2 = M[8+i];
			M[i  ] = L[0] * M_0 + L[4] * M_1 + L[8] * M_2;
			M[4+i] = L[1] * M_0 + L[5] * M_1 + L[9] * M_2;
			M[8+i] = L[2] * M_0 + L[6] * M_1 + L[10] * M_2;
		}

		tran_from = tgt_jointChanSplits[2*ti];
		rot_from = tgt_jointChanSplits[2*ti+1];
		rot_end = tgt_jointChanSplits[2*ti+2];
		for (int t = tran_from; t < rot_from; ++t) tgt_chanValues[t] = M[4*tgt_jointChans[t]+3];
		int nr = rot_end - rot_from;
		if (nr > 0) {
			const int i = (tgt_jointChans[rot_from]-3);
			const int parity = (nr == 1 ? 1 : (tgt_jointChans[rot_from+1]-i)%3);
			const int j = (i+parity)%3;
			const int k = (i+2*parity)%3;
			const float cj = sqrtf(M[4*i+i]*M[4*i+i] + M[4*j+i]*M[4*j+i]);
			if (parity != 2) {
				if (cj > 1e-10)
				{
					tgt_chanValues[rot_from] = atan2(M[4*k+j],M[4*k+k]);
					if (nr > 1) tgt_chanValues[rot_from+1] = atan2(-M[4*k+i],cj);
					if (nr > 2) tgt_chanValues[rot_from+2] = atan2(M[4*j+i],M[4*i+i]);
				}
				else
				{
					tgt_chanValues[rot_from] = atan2(-M[4*j+k],M[4*j+j]);
					if (nr > 1) tgt_chanValues[rot_from+1] = atan2(-M[4*k+i],cj);
					if (nr > 2) tgt_chanValues[rot_from+2] = 0.0;
				}
			}
			else
			{
				if (cj > 1e-10)
				{
					tgt_chanValues[rot_from] = -atan2(M[4*k+j],M[4*k+k]);
					if (nr > 1) tgt_chanValues[rot_from+1] = -atan2(-M[4*k+i],cj);
					if (nr > 2) tgt_chanValues[rot_from+2] = -atan2(M[4*j+i],M[4*i+i]);
				}
				else
				{
					tgt_chanValues[rot_from] = -atan2(-M[4*j+k],M[4*j+j]);
					if (nr > 1) tgt_chanValues[rot_from+1] = -atan2(-M[4*k+i],cj);
					if (nr > 2) tgt_chanValues[rot_from+2] = -0.0;
				}
			}
		}
	}
}

// modify Ls and chanValues so that all ball joints are in the rest position
void bake_ball_joints(object &o_Ls, object &o_jointChans, object &o_jointChanSplits, object &o_chanValues)
{
	Np3D<float>   Ls(o_Ls, -1, 3, 4);
	const int     numJoints = Ls._rows;
	Np1D<int32_t> jointChanSplits(o_jointChanSplits, 2*numJoints+1);
	Np1D<int32_t> jointChans(o_jointChans, jointChanSplits[2*numJoints]);
	Np1D<float>   chanValues(o_chanValues, -1);

	for (int ji = 0; ji < numJoints; ++ji) {
		float *M = Ls[ji];
		const int rot_from = jointChanSplits[2*ji+1];
		const int rot_end = jointChanSplits[2*ji+2];
		if (rot_end - rot_from == 3) {
			// TODO fixme, make the order of rotations be logical
			for (int ri = rot_end-1; ri >= rot_from; --ri) {
				const int ci = jointChans[ri];
				const float v = chanValues[ri];
				chanValues[ri] = 0;
				const float sv = sin(v), cv = cos(v); // sincos
				const int cj = (ci+1)%3, ck = (ci+2)%3;
				for (int i = 0; i < 12; i+= 4) {
					const float Rj = M[i+cj], Rk = M[i+ck];
					M[i+cj] = Rj*cv+Rk*sv;
					M[i+ck] = Rk*cv-Rj*sv;
				}
			}
		}
	}
}

// undo the effect of baking the ball joints, correcting the Ls and the chanValues
void unbake_ball_joints(object &o_Ls, object &o_jointChans, object &o_jointChanSplits, object &o_chanValues,
						object &o_orig_Ls)
{
	Np3D<float>   Ls(o_Ls, -1, 3, 4);
	const int     numJoints = Ls._rows;
	Np1D<int32_t> jointChanSplits(o_jointChanSplits, 2*numJoints+1);
	Np1D<int32_t> jointChans(o_jointChans, jointChanSplits[2*numJoints]);
	Np1D<float>   chanValues(o_chanValues, -1);
	Np3D<float>   orig_Ls(o_orig_Ls, numJoints, 3, 4);

	for (int ji = 0; ji < numJoints; ++ji) {
		float *M = Ls[ji];
		const float *L = orig_Ls[ji];
		// M = T L R; M_T = T + L_T; M_R = L_R R_R

		const int rot_from = jointChanSplits[2*ji+1];
		const int rot_end = jointChanSplits[2*ji+2];
		if (rot_end - rot_from == 3) {
			// TODO fixme, make the order of rotations be logical
			for (int ri = rot_end-1; ri >= rot_from; --ri) {
				const int ci = jointChans[ri];
				const float v = chanValues[ri];
				const float sv = sin(v), cv = cos(v); // sincos
				const int cj = (ci+1)%3, ck = (ci+2)%3;
				for (int i = 0; i < 12; i+= 4) {
					const float Rj = M[i+cj], Rk = M[i+ck];
					M[i+cj] = Rj*cv+Rk*sv;
					M[i+ck] = Rk*cv-Rj*sv;
				}
			}

			// M_R = L_R^T M_R
			for (int i = 0; i < 3; i++) {
				const float M_0 = M[i], M_1 = M[4+i], M_2 = M[8+i];
				M[i  ] = L[0] * M_0 + L[4] * M_1 + L[8] * M_2;
				M[4+i] = L[1] * M_0 + L[5] * M_1 + L[9] * M_2;
				M[8+i] = L[2] * M_0 + L[6] * M_1 + L[10] * M_2;
			}

			const int i = (jointChans[rot_from]-3);
			const int parity = (jointChans[rot_from+1]-i)%3;
			const int j = (i+parity)%3;
			const int k = (i+2*parity)%3;
			const float cj = sqrtf(M[4*i+i]*M[4*i+i] + M[4*j+i]*M[4*j+i]);
			if (parity != 2) {
				if (cj > 1e-10)
				{
					chanValues[rot_from] = atan2(M[4*k+j],M[4*k+k]);
					chanValues[rot_from+1] = atan2(-M[4*k+i],cj);
					chanValues[rot_from+2] = atan2(M[4*j+i],M[4*i+i]);
				}
				else
				{
					chanValues[rot_from] = atan2(-M[4*j+k],M[4*j+j]);
					chanValues[rot_from+1] = atan2(-M[4*k+i],cj);
					chanValues[rot_from+2] = 0.0;
				}
			}
			else
			{
				if (cj > 1e-10)
				{
					chanValues[rot_from] = -atan2(M[4*k+j],M[4*k+k]);
					chanValues[rot_from+1] = -atan2(-M[4*k+i],cj);
					chanValues[rot_from+2] = -atan2(M[4*j+i],M[4*i+i]);
				}
				else
				{
					chanValues[rot_from] = -atan2(-M[4*j+k],M[4*j+j]);
					chanValues[rot_from+1] = -atan2(-M[4*k+i],cj);
					chanValues[rot_from+2] = -0.0;
				}
			}
		}
		memcpy(M, L, sizeof(float)*12);
	}
}

double pose_effectors(object &o_effectors, object &o_residual,
					  object &o_Gs, object &o_effectorJoints,
					  object &o_effectorOffsets, object &o_effectorWeights,
					  object &o_effectorTargets)
{
	const Np3D<float> Gs(o_Gs, -1, 3, 4);
	const int numJoints = Gs._rows;
	const Np1D<int32_t> effectorJoints(o_effectorJoints, -1);
	const int numEffectors = effectorJoints._size;
	const Np3D<float> effectorOffsets(o_effectorOffsets, numEffectors, 3, 4);
	const Np3D<float> effectorWeights(o_effectorWeights, numEffectors, 3, 4);
	const Np3D<float> effectorTargets(o_effectorTargets, numEffectors, 3, 4);

	Np3D<float> effectors(o_effectors, numEffectors, 3, 4);
	Np3D<float> residual(o_residual, numEffectors, 3, 4);
	double sum = 0;
	for (int i = 0; i < numEffectors; ++i) {
		float *eff = effectors[i];
		float *res = residual[i];
		const int ji = effectorJoints[i];
		assert(ji >= 0 && ji < numJoints);
		const float *G = Gs[ji];
		const float *off = effectorOffsets[i];
		const float *tgt = effectorTargets[i];
		const float *wts = effectorWeights[i];
		for (int r = 0; r < 12; r+=4) {
			const float G_0 = G[r+0], G_1 = G[r+1], G_2 = G[r+2];
			eff[r+0] = G_0 * off[4*0+0] + G_1 * off[4*1+0] + G_2 * off[4*2+0];
			eff[r+1] = G_0 * off[4*0+1] + G_1 * off[4*1+1] + G_2 * off[4*2+1];
			eff[r+2] = G_0 * off[4*0+2] + G_1 * off[4*1+2] + G_2 * off[4*2+2];
			eff[r+3] = G_0 * off[4*0+3] + G_1 * off[4*1+3] + G_2 * off[4*2+3] + G[r+3];
			res[r+0] = (tgt[r+0] - eff[r+0])*wts[r+0];
			res[r+1] = (tgt[r+1] - eff[r+1])*wts[r+1];
			res[r+2] = (tgt[r+2] - eff[r+2])*wts[r+2];
			res[r+3] = (tgt[r+3] - eff[r+3])*wts[r+3];
			sum += res[r+0]*res[r+0] + res[r+1]*res[r+1] + res[r+2]*res[r+2] + res[r+3]*res[r+3];
		}
	}
	return sum;
}

double pose_effectors_single_ray(object &o_effectors, object &o_residual3, object &o_residual2, 
								 object &o_Gs, object &o_effectorJoints,
								 object &o_effectorOffsets, object &o_effectorWeights,
								 object &o_x3ds, object &o_effectorIndices_3d,
								 object &o_E, object &o_effectorIndices_2d)
{
	const Np3D<float> Gs(o_Gs, -1, 3, 4);
	const int numJoints = Gs._rows;
	const Np1D<int32_t> effectorJoints(o_effectorJoints, -1);
	const int numEffectors = effectorJoints._size;
	const Np2D<float> effectorOffsets(o_effectorOffsets, numEffectors, 3);
	const Np1D<float> effectorWeights(o_effectorWeights, numEffectors);

	const Np1D<int32_t> effectorIndices_3d(o_effectorIndices_3d, -1);
	const int         num3d = effectorIndices_3d._size;
	const Np2D<float> x3ds(o_x3ds, num3d, 3);

	const Np1D<int32_t> effectorIndices_2d(o_effectorIndices_2d, -1);
	const int         num2d = effectorIndices_2d._size;
	const Np3D<float> E(o_E, num2d, 2, 4);

	Np2D<float> effectors(o_effectors, numEffectors, 3);
	Np2D<float> residual3(o_residual3, num3d, 3);
	Np2D<float> residual2(o_residual2, num2d, 2);

	for (int li = 0; li < numEffectors; ++li) {
		float *eff = effectors[li];
		const int ji = effectorJoints[li];
		assert(ji >= 0 && ji < numJoints);
		const float *G = Gs[ji];
		const float *off = effectorOffsets[li];
		const float o0 = off[0], o1 = off[1], o2 = off[2];
		eff[0] = G[0+0] * o0 + G[0+1] * o1 + G[0+2] * o2 + G[0+3];
		eff[1] = G[4+0] * o0 + G[4+1] * o1 + G[4+2] * o2 + G[4+3];
		eff[2] = G[8+0] * o0 + G[8+1] * o1 + G[8+2] * o2 + G[8+3];
	}
	double sum = 0;
	for (int k = 0; k < num3d; ++k) {
		const int ei = effectorIndices_3d[k];
		const float *eff = effectors[ei];
		float *res = residual3[k];
		const float *tgt = x3ds[k];
		const float wt = effectorWeights[ei];
		res[0] = (tgt[0] - eff[0])*wt;
		res[1] = (tgt[1] - eff[1])*wt;
		res[2] = (tgt[2] - eff[2])*wt;
		sum += res[0]*res[0]+res[1]*res[1]+res[2]*res[2];
	}
	for (int k = 0; k < num2d; ++k) {
		float *res = residual2[k];
		const int ei = effectorIndices_2d[k];
		const float wt = effectorWeights[ei];
		const float *E_k = E[k];
		const float *eff = effectors[ei];
		const float e0 = eff[0], e1 = eff[1], e2 = eff[2];
		const float E00 = E_k[0*4+0], E01 = E_k[0*4+1], E02 = E_k[0*4+2], E03 = E_k[0*4+3];
		const float E10 = E_k[1*4+0], E11 = E_k[1*4+1], E12 = E_k[1*4+2], E13 = E_k[1*4+3];
		const double ei0 = e0*E00 + e1*E01 + e2*E02 + E03;
		const double ei1 = e0*E10 + e1*E11 + e2*E12 + E13;
		res[0] = -ei0*wt;
		res[1] = -ei1*wt;
		sum += res[0]*res[0]+res[1]*res[1];
	}
	return sum;
}

double score_effectors(object &o_Gs, object &o_effectorJoints,
					   object &o_effectorOffsets, object &o_effectorWeights,
					   object &o_effectorTargets)
{
	const Np3D<float> Gs(o_Gs, -1, 3, 4);
	const int numJoints = Gs._rows;
	const Np1D<int32_t> effectorJoints(o_effectorJoints, -1);
	const int numEffectors = effectorJoints._size;
	const Np3D<float> effectorOffsets(o_effectorOffsets, numEffectors, 3, 4);
	const Np3D<float> effectorWeights(o_effectorWeights, numEffectors, 3, 4);
	const Np3D<float> effectorTargets(o_effectorTargets, numEffectors, 3, 4);

	double sum = 0;
	for (int i = 0; i < numEffectors; ++i) {
		const int ji = effectorJoints[i];
		assert(ji >= 0 && ji < numJoints);
		const float *G = Gs[ji];
		const float *off = effectorOffsets[i];
		const float *tgt = effectorTargets[i];
		const float *wts = effectorWeights[i];
		for (int r = 0; r < 12; r+=4) {
			const float G_0 = G[r+0], G_1 = G[r+1], G_2 = G[r+2];
			double eff_0 = G_0 * off[4*0+0] + G_1 * off[4*1+0] + G_2 * off[4*2+0];
			double eff_1 = G_0 * off[4*0+1] + G_1 * off[4*1+1] + G_2 * off[4*2+1];
			double eff_2 = G_0 * off[4*0+2] + G_1 * off[4*1+2] + G_2 * off[4*2+2];
			double eff_3 = G_0 * off[4*0+3] + G_1 * off[4*1+3] + G_2 * off[4*2+3] + G[r+3];
			double res_0 = (tgt[r+0] - eff_0)*wts[r+0];
			double res_1 = (tgt[r+1] - eff_1)*wts[r+1];
			double res_2 = (tgt[r+2] - eff_2)*wts[r+2];
			double res_3 = (tgt[r+3] - eff_3)*wts[r+3];
			sum += res_0*res_0 + res_1*res_1 + res_2*res_2 + res_3*res_3;
		}
	}
	return sum;
}

object marker_positions(object &o_Gs, object &o_effectorJoints, object &o_effectorOffsets, 
					    object &o_effectorLabels, object &o_markerWeights)
{
	const Np3D<float> Gs(o_Gs, -1, 3, 4);
	const int numJoints = Gs._rows;
	const Np1D<int32_t> effectorJoints(o_effectorJoints, -1);
	const int numEffectors = effectorJoints._size;
	const Np3D<float> effectorOffsets(o_effectorOffsets, numEffectors, 3, 4);
	const Np1D<int32_t> effectorLabels(o_effectorLabels, numEffectors);
	const float *markerWeights = NULL;
	if (!o_markerWeights.is_none()) {
		const Np1D<float> tmp_markerWeights(o_markerWeights, numEffectors);
		markerWeights = tmp_markerWeights._data;
	}
	int effectors_output_size = effectorLabels.max(0)+1;
	object o_effectors(newArray2D<float>(effectors_output_size, 3));
	Np2D<float> effectors(o_effectors, effectors_output_size, 3);
	std::fill(effectors._data, effectors._data + effectors_output_size*3, 0);
	std::vector<float> scales(effectors_output_size);
	for (int i = 0; i < numEffectors; ++i) {
		const int li = effectorLabels[i];
		float *eff = effectors[li];
		const int ji = effectorJoints[i];
		assert(ji >= 0 && ji < numJoints);
		const float *G = Gs[ji];
		const float *off = effectorOffsets[i];
		const float wt = markerWeights ? markerWeights[i] : 1.0f;
		const float ox = off[3], oy = off[7], oz = off[11];
		eff[0] += (G[0]*ox + G[1]*oy + G[2]*oz + G[3]) * wt;
		eff[1] += (G[4]*ox + G[5]*oy + G[6]*oz + G[7]) * wt;
		eff[2] += (G[8]*ox + G[9]*oy + G[10]*oz + G[11]) * wt;
		scales[li] += wt;
	}
	for (int i = 0; i < effectors_output_size; ++i) {
		float *eff = effectors[i];
		float sc = (scales[i] ? 1.0f/scales[i] : 0.0f);
		eff[0] *= sc;
		eff[1] *= sc;
		eff[2] *= sc;
	}
	return o_effectors;
}

void derror_dchannel(object &o_ret,
					object &o_chanMats, object &o_usedChannels,
					object &o_usedChannelWeights,
					object &o_usedCAEs, object &o_usedCAEsSplits,
					object &o_jointChans, object &o_effectors,
					object &o_effectorWeights)
{
	const Np3D<float>   chanMats(o_chanMats, -1, 3, 4);
	const int           numChans = chanMats._rows;
	const Np1D<int32_t> usedChannels(o_usedChannels, -1);
	const int           numUsedChans = usedChannels._size;
	const Np1D<float>   usedChannelWeights(o_usedChannelWeights, numUsedChans);
	const Np1D<int32_t> usedCAEsSplits(o_usedCAEsSplits, numUsedChans+1);
	const int           numUsedCAEs = usedCAEsSplits[numUsedChans];
	const Np1D<int32_t> usedCAEs(o_usedCAEs, numUsedCAEs);
	const Np1D<int32_t> jointChans(o_jointChans, -1);
	const Np3D<float>   effectors(o_effectors, -1, 3, 4);
	const int           numEffectors = effectors._rows;
	const Np3D<float>   effectorWeights(o_effectorWeights, numEffectors, 3, 4);
	Np4D<float>         out(o_ret, numUsedChans, numEffectors, 3, 4);

	// Compute the derivative of the position of the effector with respect to each channel.
	for (int ui = 0; ui < numUsedChans; ++ui)
	{
		float *err = out[ui];
		const int ci = usedChannels[ui];
		const float cw = usedChannelWeights[ui];
		const float *RT = chanMats[ci];
		const int ct = jointChans[ci];
		for (int aei = usedCAEsSplits[ui]; aei < usedCAEsSplits[ui+1]; ++aei)
		{
			const int ei = usedCAEs[aei];
			float *ret = err + ei*12;
			const float *ewd = effectorWeights[ei];
			switch (ct) {
				case 0: case 1: case 2:
					ret[0] = ret[1] = ret[2] = ret[4] = ret[5] = ret[6] = ret[8] = ret[9] = ret[10] = 0.0f;
					ret[3] = RT[ct] * ewd[3] * cw; ret[7] = RT[ct+4] * ewd[7] * cw; ret[11] = RT[ct+8] * ewd[11] * cw; break;
				case 3: case 4: case 5:
				{
					const float *eff_ei = effectors[ei];
					const int c1 = (ct+1)%3, c2 = (ct+2)%3;
					const float RT02 = RT[c2], RT12 = RT[c2+4], RT22 = RT[c2+8];
					const float RT01 = RT[c1], RT11 = RT[c1+4], RT21 = RT[c1+8];
					const double r01 = RT02*RT11 - RT01*RT12, r02 = RT02*RT21 - RT01*RT22, r12 = RT12*RT21 - RT11*RT22;
					ret[0] = ewd[0]?(r01 * eff_ei[4] + r02 * eff_ei[8]) * ewd[0] * cw:0;
					ret[1] = ewd[1]?(r01 * eff_ei[5] + r02 * eff_ei[9]) * ewd[1] * cw:0;
					ret[2] = ewd[2]?(r01 * eff_ei[6] + r02 * eff_ei[10]) * ewd[2] * cw:0;
					ret[3] = ewd[3]?(r01 * (eff_ei[7] - RT[7]) + r02 * (eff_ei[11] - RT[11])) * ewd[3] * cw:0;
					ret[4] = ewd[4]?(-r01 * eff_ei[0] + r12 * eff_ei[8]) * ewd[4] * cw:0;
					ret[5] = ewd[5]?(-r01 * eff_ei[1] + r12 * eff_ei[9]) * ewd[5] * cw:0;
					ret[6] = ewd[6]?(-r01 * eff_ei[2] + r12 * eff_ei[10]) * ewd[6] * cw:0;
					ret[7] = ewd[7]?(-r01 * (eff_ei[3] - RT[3]) + r12 * (eff_ei[11] - RT[11])) * ewd[7] * cw:0;
					ret[8] = ewd[8]?(-r02 * eff_ei[0] - r12 * eff_ei[4]) * ewd[8] * cw:0;
					ret[9] = ewd[9]?(-r02 * eff_ei[1] - r12 * eff_ei[5]) * ewd[9] * cw:0;
					ret[10]= ewd[10]?(-r02 * eff_ei[2] - r12 * eff_ei[6]) * ewd[10] * cw:0;
					ret[11]= ewd[11]?(-r02 * (eff_ei[3] - RT[3]) - r12 * (eff_ei[7] - RT[7])) * ewd[11] * cw:0;
					break;
				}
				default: throw std::runtime_error("derror_dchannel: unexpected joint type");
			}
		}
	}
}

void derror_dchannel_single_ray(object &o_ret,
								object &o_chanMats, object &o_usedChannels,
								object &o_usedChannelWeights,
								object &o_usedCAEs, object &o_usedCAEsSplits,
								object &o_jointChans, object &o_effectors,
								object &o_effectorWeights)
{
	const Np3D<float>   chanMats(o_chanMats, -1, 3, 4);
	const int           numChans = chanMats._rows;
	const Np1D<int32_t> usedChannels(o_usedChannels, -1);
	const int           numUsedChans = usedChannels._size;
	const Np1D<float>   usedChannelWeights(o_usedChannelWeights, numUsedChans);
	const Np1D<int32_t> usedCAEsSplits(o_usedCAEsSplits, numUsedChans+1);
	const int           numUsedCAEs = usedCAEsSplits[numUsedChans];
	const Np1D<int32_t> usedCAEs(o_usedCAEs, numUsedCAEs);
	const Np1D<int32_t> jointChans(o_jointChans, -1);
	const Np2D<float>   effectors(o_effectors, -1, 3);
	const int           numEffectors = effectors._rows;
	const Np1D<float>   effectorWeights(o_effectorWeights, numEffectors);
	Np3D<float>         out(o_ret, numUsedChans, numEffectors, 3);

	// Compute the derivative of the position of the effector with respect to each channel.
	for (int ui = 0; ui < numUsedChans; ++ui)
	{
		float *err = out[ui];
		const int ci = usedChannels[ui];
		const float cw = usedChannelWeights[ui];
		const float *RT = chanMats[ci];
		const int ct = jointChans[ci];
		for (int aei = usedCAEsSplits[ui]; aei < usedCAEsSplits[ui+1]; ++aei)
		{
			const int ei = usedCAEs[aei];
			float *ret = err + ei*3;
			const float wt = cw * effectorWeights[ei];
			if (wt == 0) { ret[0] = ret[1] = ret[2] = 0; continue; }
			switch (ct) {
				case 0: case 1: case 2:
					ret[0] = RT[ct  ] * wt;
					ret[1] = RT[ct+4] * wt;
					ret[2] = RT[ct+8] * wt;
					break;
				case 3: case 4: case 5:
				{
					const float *eff_ei = effectors[ei];
					const int c1 = (ct+1)%3, c2 = (ct+2)%3;
					const float RT02 = RT[c2], RT12 = RT[c2+4], RT22 = RT[c2+8];
					const float RT01 = RT[c1], RT11 = RT[c1+4], RT21 = RT[c1+8];
					const double r01 = RT02*RT11 - RT01*RT12, r20 = RT01*RT22 - RT02*RT21, r12 = RT12*RT21 - RT11*RT22;
					const double dx = eff_ei[0] - RT[3], dy = eff_ei[1] - RT[7], dz = eff_ei[2] - RT[11];
					ret[0] = (r01 * dy - r20 * dz) * wt;
					ret[1] = (r12 * dz - r01 * dx) * wt;
					ret[2] = (r20 * dx - r12 * dy) * wt;
					break;
				}
				default: throw std::runtime_error("derror_dchannel_single_ray: unexpected joint type");
			}
		}
	}
}

/*
void derror_dchannel2(object &o_AATres, object &o_ATres, object &o_residual,
					object &o_chanMats, object &o_usedChannels,
					object &o_usedCAEs, object &o_usedCAEsSplits,
					object &o_jointChans, object &o_effectors,
					object &o_effectorWeights)
{
	const Np3D<float>   chanMats(o_chanMats, -1, 3, 4);
	const int           numChans = chanMats._rows;
	const Np1D<int32_t> usedChannels(o_usedChannels, -1);
	const int           numUsedChans = usedChannels._size;
	const Np1D<int32_t> usedCAEsSplits(o_usedCAEsSplits, numUsedChans+1);
	const int           numUsedCAEs = usedCAEsSplits[numUsedChans];
	const Np1D<int32_t> usedCAEs(o_usedCAEs, numUsedCAEs);
	const Np1D<int32_t> jointChans(o_jointChans, -1);
	const Np3D<float>   effectors(o_effectors, -1, 3, 4);
	const int           numEffectors = effectors._rows;
	const Np3D<float>   effectorWeights(o_effectorWeights, numEffectors, 3, 4);
	Np4D<float>         AATres(o_AATres, numEffectors, 3, 4);
	Np3D<float>         residual(o_residual, numEffectors, 3, 4);
	Np3D<float>         ATres(o_ATres, numUsedChans);

	// A -> A + d
	// (AT + dT) x = AT x + dT x
	// ATx += dTx
	// (A + d)(AT + dT)x = AATx + AdTx + d(ATx + dTx)
	// AATx += d ATx_post + A dTx
	// A AT x = sum(d_i) sum(d_i)T x
	// inc = sum(d_i) dT x + d sum(d_i)T x + d dT x
	// AT x = sum(d_i)T x

	// can do this computation in two passes: ATx in the first pass, AATx in the second

	// Compute the derivative of the position of the effector with respect to each channel.
	for (int ui = 0; ui < numUsedChans; ++ui)
	{
		float *ATres_ui = ATres[ui];
		const int ci = usedChannels[ui];
		const float *RT = chanMats[ci];
		const int ct = jointChans[ci];
		for (int aei = usedCAEsSplits[ui]; aei < usedCAEsSplits[ui+1]; ++aei)
		{
			const int ei = usedCAEs[aei];
			float *tmp = AAT_ui + ei*12; // tmp is d : A[ui,ei*12:ei*12+12]
			const float *ewd = effectorWeights[ei];
			switch (ct) {
				case 0: case 1: case 2:
					tmp[0] = tmp[1] = tmp[2] = tmp[4] = tmp[5] = tmp[6] = tmp[8] = tmp[9] = tmp[10] = 0.0f;
					tmp[3] = RT[ct] * ewd[3]; tmp[7] = RT[ct+4] * ewd[7]; tmp[11] = RT[ct+8] * ewd[11]; break;
				case 3: case 4: case 5:
				{
					const float *eff_ei = effectors[ei];
					const int c1 = (ct+1)%3, c2 = (ct+2)%3;
					const float RT02 = RT[c2], RT12 = RT[c2+4], RT22 = RT[c2+8];
					const float RT01 = RT[c1], RT11 = RT[c1+4], RT21 = RT[c1+8];
					const float r01 = RT02*RT11 - RT01*RT12, r02 = RT02*RT21 - RT01*RT22, r12 = RT12*RT21 - RT11*RT22;
					tmp[0] = (r01 * eff_ei[4] + r02 * eff_ei[8]) * ewd[0];
					tmp[1] = (r01 * eff_ei[5] + r02 * eff_ei[9]) * ewd[1];
					tmp[2] = (r01 * eff_ei[6] + r02 * eff_ei[10]) * ewd[2];
					tmp[3] = (r01 * (eff_ei[7] - RT[7]) + r02 * (eff_ei[11] - RT[11])) * ewd[3];
					tmp[4] = (-r01 * eff_ei[0] + r12 * eff_ei[8]) * ewd[4];
					tmp[5] = (-r01 * eff_ei[1] + r12 * eff_ei[9]) * ewd[5];
					tmp[6] = (-r01 * eff_ei[2] + r12 * eff_ei[10]) * ewd[6];
					tmp[7] = (-r01 * (eff_ei[3] - RT[3]) + r12 * (eff_ei[11] - RT[11])) * ewd[7];
					tmp[8] = (-r02 * eff_ei[0] - r12 * eff_ei[4]) * ewd[8];
					tmp[9] = (-r02 * eff_ei[1] - r12 * eff_ei[5]) * ewd[9];
					tmp[10]= (-r02 * eff_ei[2] - r12 * eff_ei[6]) * ewd[10];
					tmp[11]= (-r02 * (eff_ei[3] - RT[3]) - r12 * (eff_ei[7] - RT[7])) * ewd[11];
					break;
				}
				default: throw std::runtime_error("derror_dchannel: unexpected joint type");
			}
		}
	}
}
*/

void JTJ(object &o_JTJ, object &o_JTB, object &o_JT, object &o_B, object &o_usedEffectors)
 {
	const Np2D<float> JT(o_JT, -1, -1);
	const int         numUsedChans = JT._rows, numEffectors = JT._cols;
	const Np1D<float> B(o_B, numEffectors);
	Np2D<float>       JTJ(o_JTJ, numUsedChans, numUsedChans);
	Np1D<float>       JTB(o_JTB, numUsedChans);
	Np1D<int32_t>     usedEffectors(o_usedEffectors, -1);
	const int         numUsedEffectors = usedEffectors._size;
	for (int i = 0; i < numUsedChans; ++i) {
		const float *JT_i = JT[i];
		double sum = 0;
		for (int k = 0; k < numUsedEffectors; ++k) sum += JT_i[usedEffectors[k]]*B[usedEffectors[k]];
		JTB[i] = float(sum);
		for (int j = 0; j <= i; ++j) {
			const float *JT_j = JT[j];
			double sum = 0;
			for (int k = 0; k < numUsedEffectors; ++k) sum += JT_i[usedEffectors[k]]*JT_j[usedEffectors[k]];
			JTJ[i][j] = JTJ[j][i] = float(sum);
		}
	}
}

void JTJ_single_ray(object &o_JTJ, object &o_JTB, object &o_JT, object &o_residual3, object &o_effectorIndices_3d,
					object &o_E, object &o_effectorIndices_2d, object &o_residual2)
					{
	const Np2D<float> JT(o_JT, -1, -1);
	const int         numUsedChans = JT._rows;
	const int         numEffectors = JT._cols;
	const Np1D<int32_t> effectorIndices_3d(o_effectorIndices_3d, -1);
	const int         num3d = effectorIndices_3d._size;
	const Np2D<float> residual3(o_residual3, num3d, 3);
	const Np1D<int32_t> effectorIndices_2d(o_effectorIndices_2d, -1);
	const int         num2d = effectorIndices_2d._size;
	const Np3D<float> E(o_E, num2d, 2, 4);
	const Np2D<float> residual2(o_residual2, num2d, 2);


	//(J;EJ) delta_channels = (delta_x3d ; delta_x2d)
	// B = delta_x3d
	// E[:,:,3] = delta_x2d
	//(JT,JTET)(J;EJ)
	//(JTJ + JTETEJ) dc = JT dx3 + JTET dx2​

	// E_ij J_jk ; i<2,j<3,k<NC m<3
	// J_lm E_mi E_ij J_jk = JTETEJ_lk = sum_m3,i2,j3 J_lm E_mi E_ij J_jk
	
	Np2D<float>       JTJ(o_JTJ, numUsedChans, numUsedChans);
	Np1D<float>       JTB(o_JTB, numUsedChans);
	for (int i = 0; i < numUsedChans; ++i) {
		const float *JT_i = JT[i];
		double sum = 0;
		float *JTJ_i = JTJ[i];
		for (int j = 0; j <= i; ++j) { JTJ_i[j] = 0; }
		for (int k = 0; k < num3d; ++k) {
			const int ei = effectorIndices_3d[k]*3;
			const float *res = residual3[k];
			const float i0 = JT_i[ei], i1 = JT_i[ei+1], i2 = JT_i[ei+2];
			sum += i0*res[0] + i1*res[1] + i2*res[2];
			for (int j = 0; j <= i; ++j) {
				const float *JT_j = JT[j];
				const float j0 = JT_j[ei], j1 = JT_j[ei+1], j2 = JT_j[ei+2];
				JTJ_i[j] += i0*j0 + i1*j1 + i2*j2;
			}
		}
		for (int k = 0; k < num2d; ++k) {
			const int ei = effectorIndices_2d[k]*3;
			const float *res = residual2[k];
			const float *E_k = E[k];
			const float E00 = E_k[0*4+0], E01 = E_k[0*4+1], E02 = E_k[0*4+2];
			const float E10 = E_k[1*4+0], E11 = E_k[1*4+1], E12 = E_k[1*4+2];
			const float i0 = JT_i[ei], i1 = JT_i[ei+1], i2 = JT_i[ei+2];
			const double JTi0 = i0*E00 + i1*E01 + i2*E02;
			const double JTi1 = i0*E10 + i1*E11 + i2*E12;
			sum += JTi0*res[0] + JTi1*res[1];
			for (int j = 0; j <= i; ++j) {
				const float *JT_j = JT[j];
				const float j0 = JT_j[ei], j1 = JT_j[ei+1], j2 = JT_j[ei+2];
				const double JTj0 = j0*E00 + j1*E01 + j2*E02;
				const double JTj1 = j0*E10 + j1*E11 + j2*E12;
				JTJ_i[j] += JTi0 * JTj0 + JTi1 * JTj1;
			}
		}
		JTB[i] = float(sum);
		for (int j = 0; j < i; ++j) { JTJ[j][i] = JTJ_i[j]; }
	}
}

double J_transpose(object &o_delta, object &o_JJTB, object &o_JT, object &o_B)
 {
	const Np2D<float> JT(o_JT, -1, -1);
	const int         numUsedChans = JT._rows, numEffectors = JT._cols;
	const Np1D<float> B(o_B, numEffectors);
	Np1D<float>       delta(o_delta, numUsedChans);
	Np1D<float>       JJTB(o_JJTB, numEffectors);
	//delta = np.dot(JT, B)
	for (int r = 0; r < numUsedChans; ++r) {
		const float *JT_r = JT[r];
		const float *B_data = B._data;
		double sum = 0;
		for (int c = 0; c < numEffectors; ++c) sum += JT_r[c] * B_data[c];
		delta[r] = float(sum);
	}
	// JJTB = np.dot(JT.T,delta)
	// den = np.dot(JJTB,JJTB)+1.0
	double den = 1.0;
	for (int r = 0; r < numEffectors; ++r) {
		const float *JT_data = JT._data + r;
		const float *d_data = delta._data;
		double sum = 0;
		for (int c = 0; c < numUsedChans; ++c) sum += JT_data[c*numEffectors] * d_data[c];
		JJTB[r] = float(sum);
		den += sum*sum;
	}
	// scale = np.dot(B,JJTB)/den
	double scale = 0;
	for (int r = 0; r < numEffectors; ++r) scale += B[r] * JJTB[r];
	scale /= den;
	// delta *= scale
	for (int c = 0; c < numUsedChans; ++c) delta[c] *= float(scale);
	return scale;
}

double line_search(object &o_channelVals, object &o_usedChannels, object &o_delta,
				   object &Gs, object &Ls, object &boneParents,
				   object &jointChans, object &jointChanSplits, object &rootMat,
				   object &effectorJoints, object &effectorOffsets, object &effectorWeights,
				   object &effectorTargets,
				   int innerIts, double bestScore)
{
	Np1D<float> channelVals(o_channelVals,-1);
	float *cvs = channelVals._data;
	const int numChannels = channelVals._size;
	float *copyCVs = new float[numChannels];
	memcpy(copyCVs, cvs, numChannels*sizeof(float));
	const Np1D<int32_t> usedChannels(o_usedChannels, -1);
	int32_t *ucs = usedChannels._data;
	const int           numUsedChans = usedChannels._size;
	const Np1D<float> delta(o_delta, numUsedChans);
	float alpha = 1.0f, bestAlpha = 0.0f;
	for (int it2 = 0; it2 < innerIts; ++it2) {
		float testAlpha = alpha+bestAlpha;
		for (int i = 0; i < numUsedChans; ++i) {
			const int ci = ucs[i];
			cvs[ci] = copyCVs[ci] + delta[i]*testAlpha;
		}
		// TODO: np.clip(testCVs, channelLimits[:,0], channelLimits[:,1], out = testCVs)
		pose_skeleton(Gs, Ls, boneParents, jointChans, jointChanSplits, o_channelVals, rootMat);
		double testScore = score_effectors(Gs, effectorJoints, effectorOffsets, effectorWeights, effectorTargets);
		if (testScore < bestScore) {
			bestAlpha = testAlpha;
			bestScore = testScore;
		}
		else { alpha *= -0.7071067811865475f; } // toggle around the best
		if (bestAlpha+alpha < 0) alpha = -alpha;
	}
	for (int i = 0; i < numUsedChans; ++i) {
		const int ci = ucs[i];
		cvs[ci] = copyCVs[ci] + delta[i]*bestAlpha;
	}
	delete[] copyCVs;
	return bestScore;
}

void linsolveN3(const float *E, const int *dis, const int size, float *out)
 {
	/* Solve E0 x + e0 = 0 via (E0^T E0)x = -E0^T e0. E is Nx2x3 */
	// Equivalent code:
	// E0, e0 = E[dis,:,:3].reshape(-1,3),E[dis,:,3].reshape(-1)
	// out[:] = np.linalg.solve(np.dot(E0.T,E0)+np.eye(3)*1e-8,-np.dot(E0.T,e0))
	// Now form this equation and solve:
	//|a b c|[x y z] + |g| = 0
	//|b d e|          |h|
	//|c e f|          |i|
	 // force strictly positive det by adding 1e-8 to diagonal (a,d,f)
	double a = 1e-8, b = 0, c = 0, g = 0;
	double d = 1e-8, e = 0, h = 0;
	double f = 1e-8, i = 0;
	for (int di = 0; di < size; ++di) {
		const float *t = E + dis[di]*8;
		const float e0 = *t++;
		const float e1 = *t++;
		const float e2 = *t++;
		const float e3 = *t++;
		const float f0 = *t++;
		const float f1 = *t++;
		const float f2 = *t++;
		const float f3 = *t++;
		a += e0*e0+f0*f0; b += e0*e1+f0*f1; c += e0*e2+f0*f2; g += e0*e3+f0*f3;
		d += e1*e1+f1*f1; e += e1*e2+f1*f2; h += e1*e3+f1*f3;
		f += e2*e2+f2*f2; i += e2*e3+f2*f3;
	}
	const double be_cd = b*e-c*d;
	const double bc_ae = b*c-a*e;
	const double ce_bf = c*e-b*f;
	const double ad_bb = a*d-b*b;
	const double af_cc = a*f-c*c;
	const double df_ee = d*f-e*e;
	const double sc = -1.0/(c*be_cd + e*bc_ae + f*ad_bb);
	out[0] = float((g*df_ee + h*ce_bf + i*be_cd)*sc);
	out[1] = float((g*ce_bf + h*af_cc + i*bc_ae)*sc);
	out[2] = float((g*be_cd + h*bc_ae + i*ad_bb)*sc);
}

object compute_E(object &o_x2ds, object &o_splits, object &o_Ps)
 {
	/* Form this equation: E x = e by concatenating the constraints (two rows per ray) and solve for x (remember to divide by -z).
	[P00 + P20 px, P01 + P21 px, P02 + P22 px][x; y; z] = -[ P03 + P23 px ]
	[P10 + P20 py, P11 + P21 py, P12 + P22 py]             [ P13 + P23 py ]
	If the projection matrices are divided through by the focal length then the errors should be 3D-like (ok to mix with 3d equations)
	Use the same equations to add single ray constraints to IK: derr(x)/dc = E dx/dc; residual = E x - e.*/

	const Np2D<float> x2ds(o_x2ds, -1, 2);
	const int num2ds = x2ds._rows;
	const Np1D<int32_t> splits(o_splits, -1);
	const int numCams = splits._size-1;
	const Np3D<float> Ps(o_Ps, numCams, 3, 4);
	
	object o_E(newArray3D<float>(num2ds, 2, 4));
	Np3D<float> E(o_E, num2ds, 2, 4);
	
	// populate the ray equations for each camera
	for (int ci = 0; ci < numCams; ++ci) {
		const float *P = Ps[ci];
		const int c0 = splits[ci], c1 = splits[ci+1];
		for (int ei = c0; ei < c1; ++ei) {
			float *E0 = E[ei];
			const float *x2d = x2ds[ei];
			const float px = x2d[0], py = x2d[1];
			E0[0] = P[0] + P[ 8]*px;
			E0[1] = P[1] + P[ 9]*px;
			E0[2] = P[2] + P[10]*px;
			E0[3] = P[3] + P[11]*px;
			E0[4] = P[4] + P[ 8]*py;
			E0[5] = P[5] + P[ 9]*py;
			E0[6] = P[6] + P[10]*py;
			E0[7] = P[7] + P[11]*py;
		}
	}
	return o_E;
}

void solve_x3ds_only(
	object &o_E, int32_t *labels, bool robust, std::vector<float> &x3ds, int min_rays, Np2D<float> &rays,
	bool forceRayAgreement
	)
{
	/* Given some labelled rays, generate labelled 3d positions for every multiply-labelled point and equations for
	every single-labelled point. NB with the robust flag set, this modifies E. */
	Np3D<float> E(o_E, -1, 2, 4);
	const int32_t num2ds = E._rows;
	int numLabels = 1;
	if (num2ds > 0) numLabels = *std::max_element(labels, labels + num2ds) + 1;

	std::vector<int> counts(numLabels);
	std::vector<int32_t> x3d_labels;
	for (int di = 0; di < num2ds; ++di) 
	{
		const int li = labels[di];
		if (li != -1) counts[li]++;
	}

	// the indices of the detection for each label
	std::vector<int> label_dis(num2ds);
	for (int di = 0; di < num2ds; ++di) label_dis[di] = -1;
	std::vector<int> label_splits(numLabels+1);
	for (int cumsum = 0, li = 0; li < numLabels; ++li) 
	{
		cumsum += counts[li];
		label_splits[li+1] = cumsum;
	}

	// make a copy of the bounds; use this for counting in the rays
	std::vector<int> index(label_splits.begin(), label_splits.end());

	for (int di = 0; di < num2ds; ++di) 
	{
		const int li = labels[di];
		if (li != -1) 
		{
			label_dis[index[li]++] = di;
		}
	}

	// Check if we're doing an agreement test
	if (!forceRayAgreement)
	{
		for (int li = 0; li < numLabels; ++li) 
		{
			// find all the min_rays+ ray labels
			if (counts[li] >= min_rays) x3d_labels.push_back(li);
		}
	}
	else
	{
		for (int li = 0; li < numLabels; ++li) 
		{
			if (counts[li] < min_rays) continue;
			const int l0 = label_splits[li], l1 = label_splits[li + 1];

			std::map<int, float> agreements;
			std::map<int, std::vector<int> > goodRays;
			
			for (int lj = l0; lj < l1; ++lj)
			{
				int d_lj = label_dis[lj];
				agreements[lj] = 0;
				for (int lk = l0; lk < l1; ++lk)
				{
					if (lj == lk) continue;
					int d_lk = label_dis[lk];
					float dp = rays[d_lj][0] * rays[d_lk][0] + rays[d_lj][1] * rays[d_lk][1] + rays[d_lj][2] * rays[d_lk][2];
					if (dp > 0) goodRays[lj].push_back(d_lk);
					agreements[lj] += dp;
				}
			}

			std::map<int, float>::const_iterator bestRay = std::max_element( 
				agreements.begin(), agreements.end(), ( 
					boost::bind(&std::map<int, float>::value_type::second, _1) < 
					boost::bind(&std::map<int, float>::value_type::second, _2) 
				) 
			);

			int bestRayIdx = bestRay->first;
			if (goodRays[bestRayIdx].size() >= min_rays)
			{
				x3d_labels.push_back(li);
			}
		}
	}

	// compute the 3d points
	const int num3d_labels = x3d_labels.size();
	x3ds.resize(num3d_labels*3);
	if (num3d_labels) {
		float *x3p = &x3ds[0];
		for (int i = 0; i < num3d_labels; ++i) {
			const int li = x3d_labels[i];
			const int l0 = label_splits[li], l1 = label_splits[li+1];
			float *x = x3p+3*i;
			linsolveN3(E._data, &label_dis[l0], l1-l0, x);

			if (robust) {
				// reweight the equations by their agreement
				for (int lj = l0; lj < l1; ++lj) {
					float *t = E[label_dis[lj]];
					const float r0 = t[0]*x[0] + t[1]*x[1] + t[2]*x[2] + t[3];
					const float r1 = t[4]*x[0] + t[5]*x[1] + t[6]*x[2] + t[7];
					const float sc = 10./(10.+sqrtf(r0*r0+r1*r1));
					t[0] *= sc; t[1] *= sc; t[2] *= sc; t[3] *= sc;
					t[4] *= sc; t[5] *= sc; t[6] *= sc; t[7] *= sc;
				}
				linsolveN3(E._data, &label_dis[l0], l1-l0, x);
			}
		}
	}
}

tuple solve_x3ds_base2(float *E, int num2ds, const int32_t *labels, bool robust, int min_rays = 2)
{
	/* Given some labelled rays, generate labelled 3d positions for every multiply-labelled point and equations for
	every single-labelled point. NB with the robust flag set, this modifies E. */

	const int numLabels = num2ds ? *std::max_element(labels,labels+num2ds) + 1 : 0;

	std::vector<int> counts(numLabels);
	std::vector<int32_t> x3d_labels;
	std::vector<int32_t> x2d_labels;
	for (int i = 0; i < num2ds; ++i) {
		const int li = labels[i];
		if (li != -1) counts[li]++;
	}
	for (int li = 0; li < numLabels; ++li) {
		// find all the single ray labels
		if (counts[li] == 1) x2d_labels.push_back(li);
		// find all the 2+ ray labels
		if (counts[li] >= min_rays) x3d_labels.push_back(li);
	}
	// the indices of the detection for each label
	std::vector<int> label_dis(num2ds);
	for (int di = 0; di < num2ds; ++di) label_dis[di] = -1;
	std::vector<int> label_splits(numLabels+1);
	for (int cumsum = 0, li = 0; li < numLabels; ++li) {
		cumsum += counts[li];
		label_splits[li+1] = cumsum;
	}
	// make a copy of the bounds; use this for counting in the rays
	std::vector<int> index(label_splits.begin(), label_splits.end());
	for (int di = 0; di < num2ds; ++di) {
		const int li = labels[di];
		if (li != -1) {
			label_dis[index[li]++] = di;
		}
	}

	// compute the 3d points
	const int num3d_labels = x3d_labels.size();
	object o_x3ds(newArray2D<float>(num3d_labels, 3));
	Np2D<float> x3ds(o_x3ds, num3d_labels, 3);
	for (int i = 0; i < num3d_labels; ++i) {
		const int li = x3d_labels[i];
		const int l0 = label_splits[li], l1 = label_splits[li+1];
		float *x = x3ds[i];
		linsolveN3(E, &label_dis[l0], l1-l0, x);
		if (robust) {
			// reweight the equations by their agreement
			for (int lj = l0; lj < l1; ++lj) {
				float *t = E + (label_dis[lj])*8;
				const float r0 = t[0]*x[0] + t[1]*x[1] + t[2]*x[2] + t[3];
				const float r1 = t[4]*x[0] + t[5]*x[1] + t[6]*x[2] + t[7];
				const float sc = 10./(10.+sqrtf(r0*r0+r1*r1));
				t[0] *= sc; t[1] *= sc; t[2] *= sc; t[3] *= sc;
				t[4] *= sc; t[5] *= sc; t[6] *= sc; t[7] *= sc;
			}
			linsolveN3(E, &label_dis[l0], l1-l0, x);
		}
	}
	// compute the 2d points
	const int num2d_labels = x2d_labels.size();
	object o_x3d_labels(newArrayFromVector<int32_t>(x3d_labels));
	object o_E2(newArray3D<float>(num2d_labels, 2, 4));
	Np3D<float> E2(o_E2, num2d_labels, 2, 4);
	for (int i = 0; i < x2d_labels.size(); ++i)
	{
		const int li = x2d_labels[i];
		const int di = label_dis[label_splits[li]];
		memcpy(E2[i], E + di*8, 8*sizeof(float));
	}
	object o_x2d_labels(newArrayFromVector<int32_t>(x2d_labels));
	return make_tuple(o_x3ds, o_x3d_labels, o_E2, o_x2d_labels);
}

tuple solve_x3ds_base(object &o_x2ds, object &o_splits, object &o_labels, object &o_Ps, bool robust, int min_rays = 2)
{
	// TODO tidy this up, or delete it
	object o_E = compute_E(o_x2ds, o_splits, o_Ps);
	const Np1D<int32_t> labels(o_labels, -1);
	const int num2ds = labels._size;
	Np3D<float> E(o_E, num2ds, 2, 4);
	return solve_x3ds_base2(E._data, num2ds, labels._data, robust, min_rays);
}

tuple solve_x3ds(object &o_x2ds, object &o_splits, object &o_labels, object &o_Ps, bool robust)
{
	return solve_x3ds_base(o_x2ds, o_splits, o_labels, o_Ps, robust);
}

tuple solve_x3ds_rays(object &o_x2ds, object &o_splits, object &o_labels, object &o_Ps, bool robust, int min_rays)
{
	return solve_x3ds_base(o_x2ds, o_splits, o_labels, o_Ps, robust, min_rays);
}

void dot(object &o_A, object &o_x, object &o_y) {
	// Return y = A x
	const Np2D<float> A(o_A, -1, -1);
	const int rows = A._rows, cols = A._cols;
	const Np1D<float> x(o_x, cols);
	Np1D<float> y(o_y, rows);
	/*
	for (int r = 0; r < rows; ++r) {
		double sum = 0;
		const float *A_i = A[r];
		const float *x_i = x._data;
		for (int c = 0; c < cols; ++c) {
			sum += *A_i++ * *x_i++;
		}
		y[r] = float(sum);
	}*/
	float *y0 = y._data;
	for (int c = 0; c < cols; ++c) {
		const float *A_i = A._data + c;
		const float x_i = x[c];
		if (x_i) {
			for (int r = 0; r < rows; ++r) {
				y0[r] += *A_i * x_i;
				A_i += cols;
			}
		}
	}
}

object dets_to_rays(object &o_x2ds, object &o_splits, object &o_Ks, object &o_RTs) {
	/*Convert 2d coordinates to normalized rays.*/
	// (px,py,-1) = K R (a(x, y, z) - T)
	// (px - ox,py - oy,-f)^T R = a(x,y,z)-T
	const Np2D<float> x2ds(o_x2ds, -1, 2);
	const Np1D<int32_t> splits(o_splits,-1);
	const int numDets = x2ds._rows;
	const int numCameras = splits._size-1;
	const Np3D<float> Ks(o_Ks, numCameras, 3, 3);
	const Np3D<float> RTs(o_RTs, numCameras, 3, 4);
	assert(splits[numCameras] <= numDets);
	object o_rays(newArray2D<float>(numDets, 3));
	Np2D<float> rays(o_rays, numDets, 3);
	float *ray = rays._data;
	for (int ci = 0; ci < numCameras; ++ci) {
		const float *K = Ks[ci];
		const float K0 = K[2], K1 = K[5], K2 = -K[0];
		const float *RT = RTs[ci];
		const int c0 = splits[ci], c1 = splits[ci+1];
		for (int di = c0; di < c1; ++di) {
			const float *x2 = x2ds[di];
			const float x = x2[0] + K0, y = x2[1] + K1, z = K2;
			const float r0 = x * RT[0] + y * RT[4] + K2 * RT[8];
			const float r1 = x * RT[1] + y * RT[5] + K2 * RT[9];
			const float r2 = x * RT[2] + y * RT[6] + K2 * RT[10];
			float sc = r0*r0+r1*r1+r2*r2;
			if (sc != 0) sc = sqrt(1.0/sc);
			*ray++ = r0*sc;
			*ray++ = r1*sc;
			*ray++ = r2*sc;
		}
	}
	return o_rays;
}

inline void project_x3d(
	std::vector<float>& x2ds, 
	std::vector<int>& labels, 
	const float* const x3d, 
	const float* const P,
	const int label_x3d,
	const float* const normal_x3d = NULL
	)
{
	const float x = x3d[0], y = x3d[1], z = x3d[2];
	const double pz = P[8] * x + P[9] * y + P[10] * z + P[11];
	if (pz >= 0) return; // point must be in front of the camera (negative-z)

	if (normal_x3d)
	{
		const float nx = -normal_x3d[0], ny = -normal_x3d[1], nz = -normal_x3d[2];
		const float dp = P[8] * nx + P[9] * ny + P[10] * nz;
		if (dp < 0) return;
	}

	const double sc = -1.0 / pz;
	const float px = float((P[0] * x + P[1] * y + P[2] * z + P[3]) * sc);
	const float py = float((P[4] * x + P[5] * y + P[6] * z + P[7]) * sc);
	// because of undistortion, we might see a few points 'outside' the frame.
	// TODO maybe compute an actual bounding box
	if (px < -1.5 || px > 1.5 || py < -1.5 || py > 1.5) return;
	x2ds.push_back(px);
	x2ds.push_back(py);
	labels.push_back(label_x3d);
}

inline void project_x3d_visibility(
	std::vector<float>& x2ds,
	std::vector<int>& labels,
	const float* const x3d,
	const float* const P,
	const int label_x3d,
	const float* const normal_x3d = NULL,
	const int numTriangles = 0,
	const float* const triangles = NULL,
	const float* const T = NULL,
	const float* normals_tris = NULL,
	const float intersectionThreshold = 100.0,
	bool generateNormals = true,
	const int cameraIndex = -1,
	const int caller = -1
	)
{
	// bool log = (cameraIndex == 0 && label_x3d < 12 && caller == 1); //(label_x3d == 8);// && cameraIndex == 2);
	//if (log) std::cout << ">> project visibility: label = " << label_x3d << " | camera = " << cameraIndex << std::endl;
	const float x = x3d[0], y = x3d[1], z = x3d[2];
	const double pz = P[8] * x + P[9] * y + P[10] * z + P[11];
	int triangleHits = 0;
	if (pz >= 0) return; // point must be in front of the camera (negative-z)

	if (normal_x3d)
	{
		const float nx = -normal_x3d[0], ny = -normal_x3d[1], nz = -normal_x3d[2];
		const float dp = P[8] * nx + P[9] * ny + P[10] * nz;
		if (dp < 0) return;
	}

	if (triangles && T)
	{
		// Calculate normalised ray from the 3D point to the camera
		float ray[3];
		float a[3], b[3], c[3];
		float tn[3];
		float c_a[3], c_b[3];
		float k[3];
		float p[3];
		float p_a[3], p_b[3], p_c[3];
		float e0[3], e1[3], e2[3];
		float cp_e0[3], cp_e1[3], cp_e2[3];

		// Calculate ray from the 3D candidate point to the current camera
		ray[0] = x - T[0], ray[1] = y - T[1], ray[2] = z - T[2];
		float dist = pow(ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2], 0.5f);
		ray[0] /= dist, ray[1] /= dist, ray[2] /= dist;

		// Check if the ray intersects the triangles
		for (unsigned int i = 0; i < numTriangles; ++i)
		{
			//if (log) std::cout << "  -> ## Triangle test: " << i << "##" << std::endl;
			const float* triangle = triangles + i * 9;

			// Triangle vertices
			a[0] = triangle[0], a[1] = triangle[1], a[2] = triangle[2];
			b[0] = triangle[3], b[1] = triangle[4], b[2] = triangle[5];
			c[0] = triangle[6], c[1] = triangle[7], c[2] = triangle[8];

			// Triangle normal
			if (normals_tris && !generateNormals)
			{
				const float* cache_n = normals_tris + i * 3;
				tn[0] = cache_n[0], tn[1] = cache_n[1], tn[2] = cache_n[2];
			}
			else
			{
				c_a[0] = c[0] - a[0], c_a[1] = c[1] - a[1], c_a[2] = c[2] - a[2];
				c_b[0] = c[0] - b[0], c_b[1] = c[1] - b[1], c_b[2] = c[2] - b[2];
				tn[0] = c_a[1] * c_b[2] - c_a[2] * c_b[1];
				tn[1] = c_a[2] * c_b[0] - c_a[0] * c_b[2];
				tn[2] = c_a[0] * c_b[1] - c_a[1] * c_b[0];
				const float tnNorm = pow(tn[0] * tn[0] + tn[1] * tn[1] + tn[2] * tn[2], 0.5f);
				tn[0] /= tnNorm, tn[1] /= tnNorm, tn[2] /= tnNorm;
			}

			// Get the dot product between the camera-to-point ray and triangle
			const float dp = tn[0] * ray[0] + tn[1] * ray[1] + tn[2] * ray[2];
			//if (abs(dp) < 0.0001) continue;

			k[0] = c[0] - T[0], k[1] = c[1] - T[1], k[2] = c[2] - T[2];
			const float t = (tn[0] * k[0] + tn[1] * k[1] + tn[2] * k[2]) / dp;

			// Check if we want to omit testing this triangle
			//  - Ray is parallel to the triangle plane
			//  - Triangle is facing away from the camera ray
			// if (abs(dp) < 0.000001 || dp > 0.0)
			if (dp > 0.0 && (t < 0 || dist < t))
			{
				//if (log) std::cout << "  -> Ray is parallel to plane or facing away, ignoring this triangle." << std::endl;
				continue;
			}

			// Only proceed if the point is in front of the camera and further away from the triangle (inside check)
			// if (t < 0 || dist < t)
			// {
			// 	// if (log) std::cout << "  -> Label " << label_x3d << ": Point is not in front of camera or in front of triangle, ignoring this triangle." << std::endl;
			// 	// continue;
			// }

			// Calculate the intersection point 'p' on the triangle
			p[0] = T[0] + t * ray[0], p[1] = T[1] + t * ray[1], p[2] = T[2] + t * ray[2];

			// Calculate the cross products for the triangle edges and the edges from the intersection point and triangle points
			p_a[0] = p[0] - a[0], p_a[1] = p[1] - a[1], p_a[2] = p[2] - a[2];
			p_b[0] = p[0] - b[0], p_b[1] = p[1] - b[1], p_b[2] = p[2] - b[2];
			p_c[0] = p[0] - c[0], p_c[1] = p[1] - c[1], p_c[2] = p[2] - c[2];
			e0[0] = b[0] - a[0], e0[1] = b[1] - a[1], e0[2] = b[2] - a[2];
			e1[0] = c[0] - b[0], e1[1] = c[1] - b[1], e1[2] = c[2] - b[2];
			e2[0] = a[0] - c[0], e2[1] = a[1] - c[1], e2[2] = a[2] - c[2];
			cp_e0[0] = e0[1] * p_a[2] - e0[2] * p_a[1], cp_e0[1] = e0[2] * p_a[0] - e0[0] * p_a[2], cp_e0[2] = e0[0] * p_a[1] - e0[1] * p_a[0];
			cp_e1[0] = e1[1] * p_b[2] - e1[2] * p_b[1], cp_e1[1] = e1[2] * p_b[0] - e1[0] * p_b[2], cp_e1[2] = e1[0] * p_b[1] - e1[1] * p_b[0];
			cp_e2[0] = e2[1] * p_c[2] - e2[2] * p_c[1], cp_e2[1] = e2[2] * p_c[0] - e2[0] * p_c[2], cp_e2[2] = e2[0] * p_c[1] - e2[1] * p_c[0];

			// Calculate the dot product between the results from the cross products and the triangle normal 
			// (dividing the plane based on each edge). 
			const float dpA = cp_e0[0] * tn[0] + cp_e0[1] * tn[1] + cp_e0[2] * tn[2];
			const float dpB = cp_e1[0] * tn[0] + cp_e1[1] * tn[1] + cp_e1[2] * tn[2];
			const float dpC = cp_e2[0] * tn[0] + cp_e2[1] * tn[1] + cp_e2[2] * tn[2];

			// Reject this 3D point (classify as hidden) if it's within the triangle unless it's fairly close (inside) to triangle
			if (dpA >= 0 && dpB >= 0 && dpC >= 0)
			{
				triangleHits++;
				// if (log) std::cout << "  -> Label " << label_x3d << " intersects with triangle: " << i << " | # hits = " << triangleHits << " | dp = " << dp << " | dist = " << (dist - t) << std::endl;
				// return;
				if (dist - t > intersectionThreshold) 
				{
					// if (log) std::cout << "    -> Returning" << std::endl;
					return;
				}
				// if (triangleHits > 1)
				// {
					// if (log) std::cout << "  -> Label " << label_x3d << ": Triangle hits exceed 1, returning." << std::endl;
					// return;
				// }
				//if (log) std::cout << "  -> Intersection threshold " << intersectionThreshold << " not exceeded for label " << label_x3d << std::endl;
				// if (!normal_x3d && dp < 0.0)
				// if (dp > 0.0) 
				// {
					//if (log) std::cout << "No normal and ray facing away from triangle.. returning." << std::endl;
					// return;
				// }
			}
		}
	}
	
	const double sc = -1.0 / pz;
	const float px = float((P[0] * x + P[1] * y + P[2] * z + P[3]) * sc);
	const float py = float((P[4] * x + P[5] * y + P[6] * z + P[7]) * sc);
	// because of undistortion, we might see a few points 'outside' the frame.
	// TODO maybe compute an actual bounding box
	if (px < -1.5 || px > 1.5 || py < -1.5 || py > 1.5) return;
	//if (log) std::cout << "  -> Adding: " << label_x3d << std::endl;
	x2ds.push_back(px);
	x2ds.push_back(py);
	labels.push_back(label_x3d);
}

tuple project(object &o_x3ds, object &o_labels_x3d, object &o_Ps)
{
	/* Project all the 3d points in all the cameras. Clip to ensure that the point is in front of the camera and in the frame.*/
	Np2D<float> x3ds(o_x3ds, -1, 3);
	const int num3ds = x3ds._rows;
	Np1D<int32_t> labels_x3d(o_labels_x3d, num3ds);
	Np3D<float> Ps(o_Ps, -1, 3, 4);

	const int numCams = Ps._rows;

	std::vector<float> x2ds; x2ds.reserve(numCams*num3ds * 2);
	std::vector<int> labels; labels.reserve(numCams*num3ds);
	std::vector<int> splits(numCams + 1);

	splits[0] = 0;
	for (int ci = 0; ci < numCams; ++ci) {
		const float *P = Ps[ci];
		const float *x3d = x3ds._data;

		for (int xi = 0; xi < num3ds; ++xi, x3d += 3) {
			project_x3d(x2ds, labels, x3d, P, labels_x3d[xi]);
		}

		splits[ci + 1] = labels.size();
	}
	object o_x2ds(newArray2DFromVector<float>(x2ds, 2));
	object o_labels(newArrayFromVector<int32_t>(labels));
	object o_splits(newArrayFromVector<int32_t>(splits));
	return make_tuple(o_x2ds, o_splits, o_labels);
}

tuple project_visibility(
	object &o_x3ds, object &o_labels_x3d, object &o_Ps, 
	boost::shared_ptr<ProjectVisibility> visibility
	)
{
	Np2D<float> x3ds(o_x3ds, -1, 3);
	const int num3ds = x3ds._rows;
	Np1D<int32_t> labels_x3d(o_labels_x3d, num3ds);
	Np3D<float> Ps(o_Ps, -1, 3, 4);

	const int numCams = Ps._rows;
	std::vector<float> x2ds; x2ds.reserve(numCams*num3ds * 2);
	std::vector<int> labels; labels.reserve(numCams*num3ds);
	std::vector<int> splits(numCams + 1);

	splits[0] = 0;
	for (int ci = 0; ci < numCams; ++ci)
	{
		const float *P = Ps[ci];
		const float *x3d = x3ds._data;
		const float *normal_x3d = visibility->normals();
		const float *T = visibility->numTriangles() > 0 ? visibility->T(ci) : NULL;

		for (int xi = 0; xi < num3ds; ++xi, x3d += 3)
		{
			if (normal_x3d) normal_x3d += 3;

			if (visibility->numTriangles() > 0)
			{
				project_x3d_visibility(
					x2ds, labels, x3d, P, labels_x3d[xi], normal_x3d, visibility->numTriangles(), visibility->triangles(), T,
					visibility->triangleNormals(), visibility->intersectionThreshold(), visibility->generateNormals(), ci, 0
					);
			}
			else
			{
				project_x3d(x2ds, labels, x3d, P, labels_x3d[xi], normal_x3d);
			}
		}

		splits[ci + 1] = labels.size();
	}

	object o_x2ds(newArray2DFromVector<float>(x2ds, 2));
	object o_labels(newArrayFromVector<int32_t>(labels));
	object o_splits(newArrayFromVector<int32_t>(splits));
	return make_tuple(o_x2ds, o_splits, o_labels);
}

void update_vels(object &o_x2ds, object &o_splits, object &o_labels, 
				 object &o_prev_x2ds, object &o_prev_splits, object &o_prev_labels,
				 object &o_vels)
{
	const Np2D<float> x2ds(o_x2ds, -1, 2);
	const int num2ds = x2ds._rows;
	const Np1D<int32_t> splits(o_splits, -1);
	const int numCams = splits._size-1;
	const Np1D<int32_t> labels(o_labels, num2ds);

	const Np2D<float>   prev_x2ds(o_prev_x2ds, -1, 2);
	const int num_prev_2ds = prev_x2ds._rows;
	const Np1D<int32_t> prev_splits(o_prev_splits, numCams+1);
	const Np1D<int32_t> prev_labels(o_prev_labels, num_prev_2ds);

	Np2D<float> vels(o_vels, num2ds, 2);

	const int numPrevLabels = (num_prev_2ds == 0 ? 0 : 1+*std::max_element(prev_labels._data, prev_labels._data+num_prev_2ds));
	std::vector<int32_t> tmp(numPrevLabels);
	for (int ci = 0; ci < numCams; ++ci) {
		for (int pli = 0; pli < numPrevLabels; ++pli) tmp[pli] = -1;
		const int c0 = splits[ci], c1 = splits[ci+1];
		const int p0 = prev_splits[ci], p1 = prev_splits[ci+1];
		if (p0 == p1) {
			for (int di = c0; di < c1; ++di) vels[di][0] = vels[di][1] = 0.0;
			continue;
		}
		for (int pdi = p0; pdi < p1; ++pdi) {
			const int li = prev_labels[pdi];
			if (li != -1) tmp[li] = pdi;
		}
		for (int di = c0; di < c1; ++di) {
			const int li = labels[di];
			const int pdi = (li == -1 || li >= numPrevLabels ? -1 : tmp[li]);
			if (pdi == -1) {
				vels[di][0] = vels[di][1] = 0.0;
				continue;
			}
			vels[di][0] = x2ds[di][0] - prev_x2ds[pdi][0];
			vels[di][1] = x2ds[di][1] - prev_x2ds[pdi][1];
		}
	}
}

struct Dot {
	size_t x0,x1,y0,y1;
	double s, sx, sy, sxx, sxy, syy;
	bool operator==(const Dot& other) {return false;}
	bool operator!=(const Dot& other) {return true;}
};



void boxBlur(float* source, float* buffer, int imWidth, int imHeight, int blurRadius)
{
	/* Performs a 2D box filter with kernel size of 2 * blurRadius + 1 
	 * (i.e. blurRadius = 2 results in a 5x5 kernel)
	 *
	 * The image edge is padded by holding the edge value
	 *
	 * box blur in place
	*/

	if (blurRadius == 0) return; // degenerate case
	
	// To normalise a kernel you divide by the number of elements
	// because we're doing it as two one dimensional operations we normalise
	// by the width of the kernel first and then by the height of the kernel
	// For a box kernel Width == Height == (2 * blurRadius + 1)
	const float normalise1D = 1.0f / (2*blurRadius + 1);
	
	// 1D Box filter on each row of kernel size (2*blurRadius + 1)
	for (int i = 0; i < imHeight; ++i)
	{
		int tileIndex = i * imWidth;
		int leftIndex = tileIndex;
		int rightIndex = tileIndex + blurRadius;
		float firstValue = source[tileIndex];
		float lastValue = source[tileIndex + imWidth - 1];
		double val = (blurRadius + 1) * firstValue;
		for (int j = 0; j < blurRadius; j++) val += source[tileIndex + j];

		// Left row edge hold value
		for (int j = 0; j <= blurRadius; j++)
		{
			val += source[rightIndex++] - firstValue;
			buffer[tileIndex++] = float(val * normalise1D);
		}

		// Non-edge section of row
		for (int j = blurRadius + 1; j < imWidth - blurRadius; j++)
		{
			val += source[rightIndex++] - buffer[leftIndex++];
			buffer[tileIndex++] = float(val * normalise1D);
		}

		// Right row edge hold value
		for (int j = imWidth - blurRadius; j < imWidth; j++)
		{
			val += lastValue - source[leftIndex++];
			buffer[tileIndex++] = float(val * normalise1D);
		}
	}

	// 1D Box filter on the columns of the buffer generated in the last iteration
	// Result is a 2d Box filter on the image
	for (int i = 0; i < imWidth; ++i)
	{
		int tileIndex = i;
		int leftIndex = tileIndex;
		int rightIndex = tileIndex + blurRadius * imWidth;
		float firstValue = buffer[tileIndex];
		float lastValue = buffer[tileIndex + imWidth * (imHeight - 1)];
		double val = (blurRadius + 1) * firstValue;
		for (int j = 0; j < blurRadius; j++) val += buffer[tileIndex + j * imWidth];

		// Top edge column hold value
		for (int j = 0; j <= blurRadius; j++)
		{
			val += buffer[rightIndex] - firstValue;
			source[tileIndex] = float(val * normalise1D);
			rightIndex += imWidth;
			tileIndex += imWidth;
		}

		// Non-edge section of column
		for (int j = blurRadius + 1; j < imHeight - blurRadius; j++)
		{
			val += buffer[rightIndex] - buffer[leftIndex];
			source[tileIndex] = float(val * normalise1D);
			leftIndex += imWidth;
			rightIndex += imWidth;
			tileIndex += imWidth;
		}

		// Bottom edge column hold value
		for (int j = imHeight - blurRadius; j < imHeight; j++)
		{
			val += lastValue - buffer[leftIndex];
			source[tileIndex] = float(val * normalise1D);
			leftIndex += imWidth;
			tileIndex += imWidth;
		}
	}
}

// in-place gaussian blur computed by 1d convolutions
void gaussBlur2(float *rd_begin, float *buffer, const int w, const int h, const float sigma)
{
	// good for sigma < 3
	const int kernel_size = int(3*sigma)|1;
	std::vector<float> kd(kernel_size);
	const int k2 = kernel_size/2;
	const float esc = -0.5/(sigma*sigma);
	float sum = 0;
	for (int i = 0; i < kernel_size; ++i) sum += (kd[i] = float(exp(esc*(i-k2)*(i-k2))));
	for (int i = 0; i < kernel_size; ++i) kd[i] /= sum;
	const float *kernel = &kd[0];

	const float *rd = rd_begin;
	float *wd;
	const float *rd_end = rd_begin + (h*w - kernel_size);
	for (rd = rd_begin, wd = buffer; rd < rd_end; rd++,wd++) {
		double s1=0;
		for (int i = 0; i < kernel_size; ++i) s1 += kernel[i]*rd[i];
		*wd = float(s1);
	}
	
	const int offset = k2*(w+1);
	wd = rd_begin + offset;
	const float *wd_end = rd_begin + h*w - offset;
	for (rd = buffer; wd < wd_end; rd++,wd++) {
		double s1=0;
		for (int i = 0, j = 0; i < kernel_size; ++i,j+=w) s1 += kernel[i]*rd[j];
		*wd = float(s1);
	}
}

// in-place gaussian blur approximated by successive box blurs
void gaussBlur(float* source, float* buffer, int w, int h, float sigma)
{
	int ks = int(sigma/2);
	boxBlur(source, buffer, w, h, ks);
	boxBlur(source, buffer, w, h, ks);
	boxBlur(source, buffer, w, h, ks);
	sigma -= ks*2;
	if (sigma > 0.1) gaussBlur2(source, buffer, w, h, sigma);
}

// dot detection
object filter_image(object &o_image, int b1, int b2)
{
	Np3D<unsigned char> image(o_image, -1, -1, 3);
	const unsigned char *data = image[0]+1; // NOTE +1 means green channel
	const size_t rows = image._rows, cols = image._cols;
	object o_ret(newArray3D<unsigned char>(rows, cols, 3));
	if (rows == 0 || cols == 0) return o_ret;

	std::vector<float> blur1(rows * cols);
	std::vector<float> blur2(rows * cols);
	std::vector<float> blur3(rows * cols);
	float *bd1 = &blur1[0];
	float *bd2 = &blur2[0];
	float *bd3 = &blur3[0];
	const int chans = 3; //image._chans;
	std::vector<float> lookup(256);
	for (int i = 0; i < 256; ++i) lookup[i] = float(pow((i+20)/275.0,2.2));
	for (int i = rows*cols-1; i >= 0; --i) bd1[i] = lookup[data[i*chans]]; // extract a single-channel image
	std::copy(bd1,bd1+rows*cols,bd2);
	
	boxBlur(bd1, bd3, cols, rows, b1);
	boxBlur(bd2, bd3, cols, rows, b2);

	Np3D<unsigned char> ret(o_ret, rows, cols, 3);
	const float *rd1 = bd1;
	const float *rd2 = bd2;
	unsigned char *wd = ret[0];
	for (int row = 0; row < rows; ++row) {
		for (int col = 0; col < cols; ++col,wd+=3,++rd1,++rd2) {
			// ?????????????
			float v = (*rd1 / *rd2)*128.0;
			wd[0] = wd[1] = wd[2] = (unsigned char)std::max<float>(0.0,std::min<float>(v,255.0));
			//float v = (sqrtf(*rd1 + 20) - sqrtf(*rd2 + 20))*3.0 + 128.0;
			//wd[0] = wd[1] = wd[2] = (unsigned char)std::min<float>(std::max<float>(0.0,v),255.0);
		}
	}
	return o_ret;

}

// dot detection
std::vector<Dot> detect_bright_dots(object &o_image, unsigned char tr, unsigned char tg, unsigned char tb)
{
	Np3D<unsigned char> image(o_image, -1, -1, 3);
	std::vector<Dot> ret;
	unsigned char *data = image[0];
	const size_t rows = image._rows, cols = image._cols;
	size_t prev_rs = 0;
	for (size_t row = 0; row < rows; ++row) {
		size_t rs = prev_rs;
		for (size_t col = 0; col < cols; ) {
			if (data[0] >= tr && data[1] >= tg && data[2] >= tb) {
				// score the strip
				const size_t c0 = col;
				double s = 0,sx = 0,sxx = 0;
				while (true) {
					double ds = data[0]-tr + data[1]-tg + data[2]-tb + 1;
					s += ds;
					sx += ds*col;
					sxx += ds*col*col;
					++col;
					data += 3;
					if (col >= cols || (data[0] < tr || data[1] < tg || data[2] < tb)) break;
				}
				// find a dot to assign it to
				while (rs < ret.size() && ret[rs].x1 <= c0) {
					++rs;   // pass dots that finish before
				}
				// c0 < col, ret[rs].x0 < ret[rs].x1
				if (rs < ret.size() && ret[rs].x0 < col) { // reaches the next dot
					size_t c1 = col;
					Dot &d = ret[rs];
					while (col < cols && (data[0] >= tr && data[1] >= tg && data[2] >= tb)) { // allow for expansion
						c1 = col+1;
						double ds = data[0]-tr + data[1]-tg + data[2]-tb + 1;
						s += ds;
						sx += ds*col;
						sxx += ds*col*col;
						++col;
						data += 3;
					}
					if (c1 > d.x1 && rs+1 < ret.size() && ret[rs+1].x0 < c1) { // merger
						Dot &d2 = ret[rs+1];
						d.s += d2.s; d.sx += d2.sx; d.sxx += d2.sxx; d.sy += d2.sy; d.sxy += d2.sxy; d.syy += d2.syy;
						d.y0 = std::min(d.y0,d2.y0); d.x1 = d2.x1;
						ret.erase(ret.begin()+rs+1);
					}
					if (d.y1 != row) { d.x0 = c0; }
					d.y1 = row;

					d.x1 = c1;
					d.s += s;
					d.sx += sx;
					d.sxx += sxx;
					d.sy += s*row;
					d.sxy += sx*row;
					d.syy += s*row*row;
				}
				else { // insert a dot
					Dot d;
					d.y0 = d.y1 = row;
					d.x0 = c0;
					d.x1 = col;
					d.s = s;
					d.sx = sx;
					d.sxx = sxx;
					d.sy = s*row;
					d.sxy = sx*row;
					d.syy = s*row*row;
					ret.insert(ret.begin()+rs, d);
				}
			}
			else {
				++col;
				data += 3;
			}
		}
		size_t write = ret.size();
		size_t read = ret.size();
		while (read > prev_rs) {
			if (ret[--read].y1 == row) {
				--write;
				if (read != write) std::swap(ret[read],ret[write]);
			}
		}
		prev_rs = write;
	}
	if (data != image._data + image._rows * image._cols * image._chans) {
		throw std::runtime_error("detect_bright_dots internal exception (187)");
	}
	for (size_t i = 0; i < ret.size(); ++i) {
		Dot &d = ret[i];
		if (d.s != 0) {
			double scale = 1.0/d.s;
			d.sx *= scale;
			d.sy *= scale;
			d.sxx *= scale;
			d.sxy *= scale;
			d.syy *= scale;
			d.sxx -= d.sx*d.sx;
			d.sxy -= d.sx*d.sy;
			d.syy -= d.sy*d.sy;
			d.sx += 0.5;
			d.sy += 0.5;
		}
	}
	return ret;
}

// dot detection
std::vector<Dot> detect_bright_dots_box(object &o_image, unsigned char tr, unsigned char tg, unsigned char tb, 
	size_t start_row, size_t start_col, size_t end_row, size_t end_col)
{
	Np3D<unsigned char> image(o_image, -1, -1, 3);
	std::vector<Dot> ret;
	unsigned char *data = image[0];
	// Commented as we don't need to know the rows
	// TO DO: Check start_row, end_row, etc... to make sure that they are within the image
	const size_t image_rows = image._rows, image_cols = image._cols;
	const size_t rows = end_row, cols = end_col;

	// Offset the data by the starting row
	data += ((image_cols * start_row)) * 3;

	size_t prev_rs = 0;
	for (size_t row = start_row; row < end_row; ++row) {
		size_t rs = prev_rs;

		// Offset the data by the starting column
		data += start_col * 3;
		for (size_t col = start_col; col < end_col; ) {
			if (data[0] >= tr && data[1] >= tg && data[2] >= tb) {
				// score the strip
				const size_t c0 = col;
				double s = 0,sx = 0,sxx = 0;
				while (true) {
					double ds = data[0]-tr + data[1]-tg + data[2]-tb + 1;
					s += ds;
					sx += ds*col;
					sxx += ds*col*col;
					++col;
					data += 3;
					if (col >= cols || (data[0] < tr || data[1] < tg || data[2] < tb)) break;
				}
				// find a dot to assign it to
				while (rs < ret.size() && ret[rs].x1 <= c0) {
					++rs;   // pass dots that finish before
				}
				// c0 < col, ret[rs].x0 < ret[rs].x1
				if (rs < ret.size() && ret[rs].x0 < col) { // reaches the next dot
					size_t c1 = col;
					Dot &d = ret[rs];
					while (col < cols && (data[0] >= tr && data[1] >= tg && data[2] >= tb)) { // allow for expansion
						c1 = col+1;
						double ds = data[0]-tr + data[1]-tg + data[2]-tb + 1;
						s += ds;
						sx += ds*col;
						sxx += ds*col*col;
						++col;
						data += 3;
					}
					if (c1 > d.x1 && rs+1 < ret.size() && ret[rs+1].x0 < c1) { // merger
						Dot &d2 = ret[rs+1];
						d.s += d2.s; d.sx += d2.sx; d.sxx += d2.sxx; d.sy += d2.sy; d.sxy += d2.sxy; d.syy += d2.syy;
						d.y0 = std::min(d.y0,d2.y0); d.x1 = d2.x1;
						ret.erase(ret.begin()+rs+1);
					}
					if (d.y1 != row) { d.x0 = c0; }
					d.y1 = row;

					d.x1 = c1;
					d.s += s;
					d.sx += sx;
					d.sxx += sxx;
					d.sy += s*row;
					d.sxy += sx*row;
					d.syy += s*row*row;
				}
				else { // insert a dot
					Dot d;
					d.y0 = d.y1 = row;
					d.x0 = c0;
					d.x1 = col;
					d.s = s;
					d.sx = sx;
					d.sxx = sxx;
					d.sy = s*row;
					d.sxy = sx*row;
					d.syy = s*row*row;
					ret.insert(ret.begin()+rs, d);
				}
			}
			else {
				++col;
				data += 3;
			}
		}
		size_t write = ret.size();
		size_t read = ret.size();
		while (read > prev_rs) {
			if (ret[--read].y1 == row) {
				--write;
				if (read != write) std::swap(ret[read],ret[write]);
			}
		}
		prev_rs = write;

		// Offset the data by the ending column
		data += (image_cols - end_col) * 3;
	}
	//if (data != image._data + (end_row - start_row) * (end_col - start_col) * image._chans) {
	//	throw std::runtime_error("detect_bright_dots internal exception (188)");
	//}
	for (size_t i = 0; i < ret.size(); ++i) {
		Dot &d = ret[i];
		if (d.s != 0) {
			double scale = 1.0/d.s;
			d.sx *= scale;
			d.sy *= scale;
			d.sxx *= scale;
			d.sxy *= scale;
			d.syy *= scale;
			d.sxx -= d.sx*d.sx;
			d.sxy -= d.sx*d.sy;
			d.syy -= d.sy*d.sy;
			d.sx += 0.5;
			d.sy += 0.5;
		}
	}
	return ret;
}


double _min_assignment_sparse(const int32_t N, 
							  const float *wMat, const int32_t *wMatI,
							  const int32_t *wMatS, const float &threshold,
							  int *u2v)
{
	const int32_t M = (wMatS[N] == 0 ? 1 : 2+*std::max_element(wMatI, wMatI+wMatS[N]));
	const int32_t NONE    = -1;
	const int32_t REMOVED = -2;
	const int32_t NOLABEL = M-1;
	const int32_t minNM  = std::min<int32_t>(N,M);
	//wMat  = [weights[wMatS[si]:wMatS[si+1]] for si in range(N)];
	//wMatI = [weightsIndices[wMatS[si]:wMatS[si+1]] for si in range(N)];
	//wZips = [zip(wMatI[si],wMat[si]) for si in range(N)];

	// allocate some workspace
	char *data = new char[(4*M+N)*sizeof(int32_t)+(2*M+2*N)*sizeof(float)];
	int32_t *v2u   = reinterpret_cast<int32_t*>(data);   // just the inverse mapping of u2v
	int32_t *v2pu  = (v2u+M);  // u2v and v2pu gives the alternating path
	int32_t *v2i   = (v2pu+M); // the inverse mapping of Vs; NONE if not added, REMOVED if i < ti
	int32_t *Us    = (v2i+M);
	int32_t *Vs    = (Us+N);
	float   *lu    = reinterpret_cast<float*>(Vs+M);
	float   *lv    = (lu+N);
	float   *slack = (lv+M);
	float   *offs  = (slack+M);
	for (int32_t i = 0; i < M; ++i) { v2u[i] = NONE; lv[i] = 0; } // these need initialising

	for (int32_t u = 0; u < N; ++u) { // greedy initialiser
		const int32_t r0 = wMatS[u], r1 = wMatS[u+1];
		if (r0 == r1) { lu[u]=threshold; u2v[u] = NOLABEL; continue; }
		int32_t si = r0; for (int32_t i = r0+1; i < r1; ++i) if (wMat[i]<wMat[si]) si = i;
		const int32_t v  = wMatI[si];
		lu[u] = wMat[si];
		if (lu[u] > threshold) { lu[u]=threshold; u2v[u] = NOLABEL; continue; } // just check for craziness
		const int32_t oldu = v2u[v];
		if (oldu == NONE) { v2u[v] = u; u2v[u] = v; } // assign
		else if (lu[oldu] > lu[u]) { v2u[v] = u; u2v[u] = v; u2v[oldu] = NONE; } // assign and bump
		else { u2v[u] = NONE; }
	}
	for (int32_t ui = 0; ui < N; ++ui) {
		int32_t u = ui;
		if (u2v[u] != NONE) { continue; } // skip if already in the matching
		float slack_offset  = -lu[u];
		const int32_t r0 = wMatS[u], r1 = wMatS[u+1]; //Vinit = wMatI[u];
		int32_t size = r1-r0+1;
		for (int32_t i = 0; i < M; ++i) v2i[i] = NONE; // v2i[:] = NONE;
		Vs[0] = NOLABEL;
		memcpy(Vs+1,wMatI+r0,(r1-r0)*sizeof(int32_t)); //Vs[1:size] = Vinit;
		v2i[NOLABEL] = 0;
		for (int ri = r0; ri < r1; ++ri) v2i[wMatI[ri]] = ri-r0+1; //v2i[Vinit] = range(1,size);
		slack[0] = -threshold;
		for (int ri = r0; ri < r1; ++ri) slack[ri-r0+1] = lv[wMatI[ri]] - wMat[ri]; //slack[1:size] = lv[Vinit] - wMat[u]; // slack[:]-slack_offset=lu[u0]+lv[:]-wMat[u0,:]
		v2pu[NOLABEL] = u;
		for (int ri = r0; ri < r1; ++ri) v2pu[wMatI[ri]] = u; // v2pu[Vinit] = u; // v2pu[v] = tree-parent of v (in Us)
		for (int32_t ti = 0; ti < minNM; ++ti) {
			assert(ti != N);
			int32_t slack_argmax = ti; // select edge (u,v) with u in Us, !Vs[v] and max slack
			for (int32_t i = ti+1; i < size; ++i) if (slack[i] > slack[slack_argmax]) slack_argmax = i;
			int32_t v             = Vs[slack_argmax];
			Vs[slack_argmax]      = Vs[ti];
			v2i[Vs[slack_argmax]] = v2i[v];
			std::swap(slack[ti],slack[slack_argmax]);
			float slack_max       = slack[ti];
			Us[ti]   = u;
			Vs[ti]   = v;
			offs[ti] = slack_max-slack_offset;
			v2i[v]   = REMOVED;
			u = v2u[v];
			if (u == NONE) {
				for (int32_t si = ti; si > 0; --si) { offs[si-1] += offs[si]; }
				for (int32_t si = 0; si <= ti; ++si) { lu[Us[si]] -= offs[si]; } //lu[Us[:ti+1]] -= offs[:ti+1];
				for (int32_t si = 0; si < ti; ++si) { lv[Vs[si]] += offs[si+1]; }//lv[Vs[:ti  ]] += offs[1:ti+1];
				while (v != NONE) { // shift along the alternating path from v to the root of the tree.
					u = v2u[v] = v2pu[v];
					std::swap(u2v[u],v);
				}
				v2u[NOLABEL] = NONE; // anyone can take this label
				break;
			}
			slack_offset = slack_max;
			const float suv_offset = lu[u]+slack_max;
			const int32_t r0 = wMatS[u], r1 = wMatS[u+1];
			for (int32_t ri = r0; ri < r1; ++ri) { // for (vi,w in zip(wMatI[u],wMat[u])) {
				const int32_t vi = wMatI[ri];
				const float w = wMat[ri];
				const int32_t si = v2i[vi];
				if (si != REMOVED) {
					const float suv = lv[vi]-w+suv_offset;
					if (si == NONE) {
						Vs[size]    = vi;
						v2i[vi]     = size;
						slack[size] = suv;
						v2pu[vi]    = u;
						size++;
					}
					else if (slack[si] < suv) { slack[si] = suv; v2pu[vi] = u; }
				}
			}
			//assert(v2i[NOLABEL] != REMOVED && v2i[NOLABEL] != NONE)
			const float suv = -threshold+suv_offset;
			if (slack[v2i[NOLABEL]] < suv) { slack[v2i[NOLABEL]] = suv; v2pu[NOLABEL] = u; }
		}
	}
	for (int32_t i = 0; i < N; ++i) if (u2v[i] == NOLABEL) u2v[i] = NONE;
	double ret = 0;
	for (int32_t i = 0; i < N; ++i) ret += lu[i];
	for (int32_t i = 0; i < M; ++i) ret += lv[i];
	delete[] data;
	return ret;
}


double min_assignment_sparse(object &o_weights, object &o_weightsIndices, object &o_weightsSplits, const float threshold,
							 object &o_u2v)
{
	/*Solve the assignment problem; find u2v s.t. u2v[u_i] != u2v[u_j] and sum(weightMatrix[u_i,u2v[u_i]]) is minimised.
	The weight matrix is sparse, with indices I[S_i:S_i+1] and weights W[S_i:S_i+1]. The Is must not include duplicates.
	Any u may also be left unassigned, with penalty weight threshold.*/
	Np1D<int32_t> u2v(o_u2v,-1);
	const int32_t N = u2v._size;
	const Np1D<float>   wMat(o_weights,-1);
	const int32_t size = wMat._size;
	const Np1D<int32_t> wMatI(o_weightsIndices, size);
	const Np1D<int32_t> wMatS(o_weightsSplits, N+1);
	assert(wMatS[N] == size);
	return _min_assignment_sparse(N, wMat._data, wMatI._data, wMatS._data, threshold, u2v._data);
}

class HashCloud2D
{
	struct HI {
		int32_t hash;
		int32_t index;
		bool operator<(const HI &that) const { return ((hash < that.hash) || (hash == that.hash && index < that.index)); }
	};

	int32_t      _num2ds;
	float       *_x2ds;
	float        _threshold;
	HI          *_hi;
	int32_t      _splits[0x401];
public:
	/*Convert 2D points into a hashcloud via a threshold.
	Return values: hs are the the hash values of the original points. horder gives the index of the 2D points,
	in order of hash value. h_splits gives the range of points in horder that share each hash value.*/
	HashCloud2D(object &o_x2ds, float threshold)
	{
		Np2D<float> x2ds(o_x2ds,-1,2);
		_num2ds = x2ds._rows;
		_x2ds = new float[_num2ds*2];
		memcpy(_x2ds,x2ds._data,_num2ds*2*sizeof(float));
		_threshold = threshold;
		_hi = new HI[_num2ds];
		hash_10(x2ds._data, _num2ds, 0, reinterpret_cast<int32_t*>(_hi));
		std::sort(_hi, _hi+_num2ds);
		for (int i = 0, j = 0; i < 0x401; ++i) {
			while (j < _num2ds && _hi[j].hash < i) ++j;
			_splits[i] = j;
		}
	}

	HashCloud2D(const float *x2ds, int32_t num2ds, float threshold)
	{
		_num2ds = num2ds;
		_x2ds = new float[_num2ds*2];
		memcpy(_x2ds,x2ds,_num2ds*2*sizeof(float));
		_threshold = threshold;
		_hi = new HI[_num2ds];
		hash_10(_x2ds, _num2ds, 0, reinterpret_cast<int32_t*>(_hi));
		std::sort(_hi, _hi+_num2ds);
		for (int i = 0, j = 0; i < 0x401; ++i) {
			while (j < _num2ds && _hi[j].hash < i) ++j;
			_splits[i] = j;
		}
	}

	virtual ~HashCloud2D() {
		delete[] _x2ds;
		delete[] _hi;
	}

	/*A 10-bit hashing function. Each point is put at the centre of a unit square, rounded down (floor),
	and converted to an integer by scaling, adding and modulus.
	Then, for any square that includes the centre point, the hash of one of its corners will match.
	Suitable for up to 100 points (expect hash collisions).*/
	void hash_10(const float *x2ds, const int32_t N, const float offset, int32_t *hi)
	{
		const float sc = 0.5/_threshold;
		for (int i = 0; i < N; ++i) {
			hi[2*i] = (int32_t(floor((x2ds[2*i]-offset)*sc))+0x20*int32_t(floor((x2ds[2*i+1]-offset)*sc)))&0x3ff;
			hi[2*i+1] = i;
		}
	}

	double threshold() { return _threshold; }
	double sq_threshold() { return _threshold*_threshold; }
	
	void _score(const float *x2ds, const int32_t size,
				std::vector<float> &scores, std::vector<int32_t> &matches, int32_t *matches_splits)
				{
		HI *hi = new HI[size];
		hash_10(x2ds, size, _threshold, reinterpret_cast<int32_t*>(hi));
		const int32_t offs[4] = { 0, 0x1, 0x20, 0x21 };
		matches_splits[0] = 0;
		const float sq_threshold = _threshold*_threshold;
		for (int32_t xi = 0; xi < size; ++xi) {
			const float sx = x2ds[xi*2], sy = x2ds[xi*2+1];
			HI &h = hi[xi];
			const int32_t base = h.hash;
			for (int32_t oi = 0; oi < 4; ++oi) {
				const int32_t hash = (base+offs[oi])&0x3ff;
				const int32_t r0 = _splits[hash], r1 = _splits[hash+1];
				for (int32_t ri = r0; ri < r1; ++ri) {
					const int32_t match = _hi[ri].index;
					const float dx = sx-_x2ds[match*2], dy = sy-_x2ds[match*2+1];
					const float score = dx*dx+dy*dy;
					if (score < sq_threshold) {
						matches.push_back(match);
						scores.push_back(score);
					}
				}
			}
			matches_splits[xi+1] = int32_t(matches.size());
		}
		delete[] hi;
	}

	void _score_nearest_N(const int32_t N, const float *x2ds, const int32_t size,
						  std::vector<float> &scores, std::vector<int32_t> &matches, int32_t *matches_splits)
						  {
		// find, for each point, the closest at most N points
		HI *hi = new HI[size];
		hash_10(x2ds, size, _threshold, reinterpret_cast<int32_t*>(hi));
		const int32_t offs[4] = { 0, 0x1, 0x20, 0x21 };
		matches_splits[0] = 0;
		const float sq_threshold = _threshold*_threshold;
		for (int32_t xi = 0; xi < size; ++xi) {
			float worstScore = -1;
			int32_t worstIndex = -1;
			int32_t count = 0;
			const float sx = x2ds[xi*2], sy = x2ds[xi*2+1];
			HI &h = hi[xi];
			const int32_t base = h.hash;
			for (int32_t oi = 0; oi < 4; ++oi) {
				const int32_t hash = (base+offs[oi])&0x3ff;
				const int32_t r0 = _splits[hash], r1 = _splits[hash+1];
				for (int32_t ri = r0; ri < r1; ++ri) {
					const int32_t match = _hi[ri].index;
					const float dx = sx-_x2ds[match*2], dy = sy-_x2ds[match*2+1];
					const float score = dx*dx+dy*dy;
					if (score < sq_threshold) {
						if (count < N) { // keep the first N
							if (score > worstScore) {
								worstIndex = matches.size();
								worstScore = score;
							}
							matches.push_back(match);
							scores.push_back(score);
							count++;
						}
						else if (score < worstScore) { // replace
							matches[worstIndex] = match;
							scores[worstIndex] = score;
							const int32_t ms = matches.size();
							worstScore = -1;
							worstIndex = -1;
							for (int pi = ms-N; pi < ms; ++pi) {
								const float score = scores[pi];
								if (score > worstScore) {
									worstIndex = pi;
									worstScore = score;
								}
							}
						}
					}
				}
			}
			matches_splits[xi+1] = int32_t(matches.size());
		}
		delete[] hi;
	}
	
	/* Given 2D points, find all the possible matches in the hashcloud.
	* Consider each point as the centre of a unit square; test the hashes of the four corners to find
	* all candidates (-threshold<dx,dy<threshold). Deal with any hash collisions.
	* The return value is equivalent to a scipy.csr_matrix.*/
	tuple score(object &o_x2ds)
	{
		const Np2D<float> x2ds(o_x2ds,-1,2);
		const int32_t size = x2ds._rows;
		std::vector<int32_t> matches; matches.reserve(size);
		std::vector<float> scores; scores.reserve(size);
		object o_matches_splits(newArray<int32_t>(size+1));
		Np1D<int32_t> matches_splits(o_matches_splits, size+1);
		_score(x2ds._data, size, scores, matches, matches_splits._data);
		object o_matches(newArray<int32_t>(matches.size()));
		object o_scores(newArray<float>(scores.size()));
		Np1D<int32_t> matches2(o_matches, matches.size());
		Np1D<float> scores2(o_scores, scores.size());
		if (matches.size()) memcpy(matches2._data, &matches[0], matches.size()*sizeof(int32_t));
		if (scores.size()) memcpy(scores2._data, &scores[0], scores.size()*sizeof(float));
		//return scores,matches,matches_splits
		return make_tuple(o_scores, o_matches, o_matches_splits);
	}
};

class HashCloud2DList
{
	const Np2D<float>   _x2ds;
	const Np1D<int32_t> _splits;
	const int           _num2ds;
	const int           _numCams;
	std::vector<HashCloud2D*> _clouds;
public:
	HashCloud2DList(object &o_x2ds, object &o_splits, float x2d_threshold) :
			_x2ds(o_x2ds,-1,2), _splits(o_splits,-1),
			_num2ds(_x2ds._rows), _numCams(_splits._size-1)
	{
		if (_splits[_numCams] > _x2ds._rows) {
			throw std::runtime_error("HashCloud2DList: invalid splits");
		}
		for (int i = 0; i < _numCams; ++i) {
			const int32_t x0 = _splits[i], x1 = _splits[i+1];
			_clouds.push_back(new HashCloud2D(x0==x1?0:_x2ds[x0], x1-x0, x2d_threshold));
		}
	}
	virtual ~HashCloud2DList()
	{
		for (int i = 0; i < _numCams; ++i) delete _clouds[i];
	}

	/* Given labelled 2d points, figure out the labels of the detections. */
	float _assign(int num_prev_x2ds, const float *prev_x2ds, const int32_t *prev_splits, const int32_t *prev_labels, float threshold,
		std::vector<int32_t> &labels, std::vector<float> *p_vels = NULL) {
		const float sq_threshold = threshold*threshold;
		std::vector<int32_t> labels_tmp(num_prev_x2ds);
		float sc = 0;
		std::vector<int32_t> matches; matches.reserve(_num2ds);
		std::vector<float> scores; scores.reserve(_num2ds);
		std::vector<int32_t> matches_splits;
		labels.resize(_num2ds);
		std::fill(labels.begin(), labels.end(), -1);
		if (p_vels) {
			p_vels->resize(_num2ds*2);
			std::fill(p_vels->begin(), p_vels->end(), 0.0);
		}
		for (int ci = 0; ci < _numCams; ++ci) {
			HashCloud2D *cloud = _clouds[ci];
			const int c0 = _splits[ci], c1 = _splits[ci+1];
			const int p0 = prev_splits[ci], p1 = prev_splits[ci+1];
			const int32_t pn = p1-p0;
			matches.clear(); scores.clear(); matches_splits.resize(pn+1);
			cloud->_score(pn==0?0:prev_x2ds+2*p0, pn, scores, matches, &matches_splits[0]);
			const int size = scores.size();
			sc += _min_assignment_sparse(pn, size?&scores[0]:0, size?&matches[0]:0, &matches_splits[0], sq_threshold, pn?&labels_tmp[p0]:0);
			for (int pi = p0; pi < p1; ++pi) {
				int32_t li = labels_tmp[pi];
				if (li == -1) continue;
				li += c0;
				assert(li < c1);
				labels[li] = prev_labels[pi];
				if (p_vels) {
					(*p_vels)[li*2  ] = _x2ds[li][0] - prev_x2ds[2*pi];
					(*p_vels)[li*2+1] = _x2ds[li][1] - prev_x2ds[2*pi+1];
				}
			}
		}
		return sc;
	}

	tuple assign(object &o_prev_x2ds, object &o_prev_splits, object &o_prev_labels, float threshold)
	{
		const Np2D<float>   prev_x2ds(o_prev_x2ds, -1, 2);
		const int num_prev_x2ds = prev_x2ds._rows;
		const Np1D<int32_t> prev_splits(o_prev_splits, _numCams+1);
		const Np1D<int32_t> prev_labels(o_prev_labels, num_prev_x2ds);
		std::vector<int32_t> labels;
		std::vector<float> vels;
		float sc = _assign(num_prev_x2ds, prev_x2ds._data, prev_splits._data, prev_labels._data, threshold, labels, &vels);
		object o_labels(newArrayFromVector<int32_t>(labels));
		object o_vels(newArray2DFromVector<float>(vels,2));
		return make_tuple(sc,o_labels,o_vels);
	}
	
	/* Given labelled 2d points, figure out the labels of the at most nearest N detections. */
	tuple assign_nearest_N(const int32_t N, object &o_prev_x2ds, object &o_prev_splits, object &o_prev_labels, float threshold)
	{
		const float sq_threshold = threshold*threshold;
		const Np2D<float>   prev_x2ds(o_prev_x2ds, -1, 2);
		const int num_prev_x2ds = prev_x2ds._rows;
		const Np1D<int32_t> prev_splits(o_prev_splits, _numCams+1);
		const Np1D<int32_t> prev_labels(o_prev_labels, num_prev_x2ds);

		std::vector<int32_t> labels_tmp(num_prev_x2ds);
		float sc = 0;
		std::vector<int32_t> matches; matches.reserve(_num2ds);
		std::vector<float> scores; scores.reserve(_num2ds);
		std::vector<int32_t> matches_splits;
		object o_labels(newArray<int32_t>(_num2ds));
		object o_vels(newArray2D<float>(_num2ds,2));
		Np1D<int32_t>       labels(o_labels, _num2ds);
		Np2D<float>         vels(o_vels, _num2ds, 2);
		for (int i = 0; i < _num2ds; ++i) {
			labels[i] = -1;
			vels[i][0] = vels[i][1] = 0.0;
		}
		for (int ci = 0; ci < _numCams; ++ci) {
			HashCloud2D *cloud = _clouds[ci];
			const int c0 = _splits[ci], c1 = _splits[ci+1];
			const int p0 = prev_splits[ci], p1 = prev_splits[ci+1];
			const int32_t pn = p1-p0;
			matches.clear(); scores.clear(); matches_splits.resize(pn+1);
			cloud->_score_nearest_N(N, pn==0?0:prev_x2ds[p0], pn, scores, matches, &matches_splits[0]);
			const int size = scores.size();
			sc += _min_assignment_sparse(pn, size?&scores[0]:0, size?&matches[0]:0, &matches_splits[0], sq_threshold, pn?&labels_tmp[p0]:0);
			for (int pi = p0; pi < p1; ++pi) {
				int32_t li = labels_tmp[pi];
				if (li == -1) continue;
				li += c0;
				assert(li < c1);
				labels[li] = prev_labels[pi];
				vels[li][0] = _x2ds[li][0] - prev_x2ds[pi][0];
				vels[li][1] = _x2ds[li][1] - prev_x2ds[pi][1];
			}
		}
		return make_tuple(sc,o_labels,o_vels);
	}

	/* Given labelled 2d points, figure out the labels of the detections. */
	tuple assign_with_vel(object &o_prev_x2ds, object &o_prev_vels, object &o_prev_splits, object &o_prev_labels, float threshold)
	{
		const float sq_threshold = threshold*threshold;
		const Np2D<float>   prev_x2ds(o_prev_x2ds, -1, 2);
		const int num2ds_prev = prev_x2ds._rows;
		const Np2D<float>   prev_vels(o_prev_vels, num2ds_prev, 2);
		const Np1D<int32_t> prev_splits(o_prev_splits, _numCams+1);
		const Np1D<int32_t> prev_labels(o_prev_labels, num2ds_prev);

		std::vector<float> pred_x2ds(num2ds_prev*2);
		for (int i = 0; i < num2ds_prev; ++i) {
			pred_x2ds[2*i+0] = prev_x2ds[i][0] + prev_vels[i][0];
			pred_x2ds[2*i+1] = prev_x2ds[i][1] + prev_vels[i][1];
		}

		std::vector<int32_t> labels_tmp(num2ds_prev);
		float sc = 0;
		std::vector<int32_t> matches; matches.reserve(_num2ds);
		std::vector<float> scores; scores.reserve(_num2ds);
		std::vector<int32_t> matches_splits;
		object o_labels(newArray<int32_t>(_num2ds));
		object o_vels(newArray2D<float>(_num2ds,2));
		Np1D<int32_t>       labels(o_labels, _num2ds);
		Np2D<float>         vels(o_vels, _num2ds, 2);
		for (int i = 0; i < _num2ds; ++i) {
			labels[i] = -1;
			vels[i][0] = vels[i][1] = 0.0;
		}
		for (int ci = 0; ci < _numCams; ++ci) {
			HashCloud2D *cloud = _clouds[ci];
			const int c0 = _splits[ci], c1 = _splits[ci+1];
			const int p0 = prev_splits[ci], p1 = prev_splits[ci+1];
			const int32_t pn = p1-p0;
			matches.clear(); scores.clear(); matches_splits.resize(pn+1);
			cloud->_score(pn?&pred_x2ds[2*p0]:0, pn, scores, matches, &matches_splits[0]);
			const int size = scores.size();
			for (int mi = 0; mi < pn; ++mi) {
				if (prev_vels[p0+mi][0] == 0.0) continue; // only tweak the weights of guys with velocity
				const int m0 = matches_splits[mi], m1 = matches_splits[mi+1];
				for (int i = m0; i < m1; ++i) {
					const int li = matches[i];
					if (li != -1) scores[i] *= 4.0;
				}
			}
			sc += _min_assignment_sparse(pn, size?&scores[0]:0, size?&matches[0]:0, &matches_splits[0], sq_threshold, pn?&labels_tmp[p0]:0);
			for (int pi = p0; pi < p1; ++pi) {
				int32_t li = labels_tmp[pi];
				if (li == -1) continue;
				li += c0;
				assert(li < c1);
				labels[li] = prev_labels[pi];
				vels[li][0] = _x2ds[li][0] - prev_x2ds[pi][0];
				vels[li][1] = _x2ds[li][1] - prev_x2ds[pi][1];
			}
		}
		return make_tuple(sc,o_labels,o_vels);
	}

	/* Given labels for the detections, and some new 2d points, figure out the labels of the new 2d points. Untested. */
	tuple propagate(object &o_labels, object &o_prev_x2ds, object &o_prev_splits)
	{
		const Np2D<float>   prev_x2ds(o_prev_x2ds, -1, 2);
		const int num2ds_prev = prev_x2ds._rows;
		const Np1D<int32_t> prev_splits(o_prev_splits, _numCams+1);
		const Np1D<int32_t> labels(o_labels, _num2ds);

		std::vector<int32_t> labels_tmp(prev_x2ds._rows);
		float sc = 0;
		std::vector<int32_t> matches; matches.reserve(_num2ds);
		std::vector<float> scores; scores.reserve(_num2ds);
		std::vector<int32_t> matches_splits;
		object o_prev_labels(newArray<int32_t>(num2ds_prev));
		object o_prev_vels(newArray2D<float>(num2ds_prev,2));
		Np1D<int32_t>       prev_labels(o_prev_labels, _num2ds);
		Np2D<float>         prev_vels(o_prev_vels, _num2ds, 2);
		for (int i = 0; i < num2ds_prev; ++i) {
			prev_labels[i] = -1;
			prev_vels[i][0] = prev_vels[i][1] = 0.0;
		}
		for (int ci = 0; ci < _numCams; ++ci) {
			HashCloud2D *cloud = _clouds[ci];
			const int c0 = _splits[ci], c1 = _splits[ci+1];
			const int p0 = prev_splits[ci], p1 = prev_splits[ci+1];
			const int32_t pn = p1-p0;
			matches.clear(); scores.clear(); matches_splits.resize(pn+1);
			cloud->_score(pn==0?0:prev_x2ds[p0], pn, scores, matches, &matches_splits[0]);
			sc += _min_assignment_sparse(pn, scores.size()?&scores[0]:0, matches.size()?&matches[0]:0, &matches_splits[0], cloud->sq_threshold(), pn==0?0:&labels_tmp[p0]);
			for (int pi = p0; pi < p1; ++pi) {
				int32_t li = labels_tmp[pi];
				if (li == -1) continue;
				li += c0;
				assert(li < c1);
				prev_labels[pi] = labels[li];
				prev_vels[pi][0] = prev_x2ds[pi][0] - _x2ds[li][0];
				prev_vels[pi][1] = prev_x2ds[pi][1] - _x2ds[li][1];
			}
		}
		return make_tuple(sc,o_prev_labels,o_prev_vels);
	}

	tuple project_assign(object &o_x3ds, object &o_labels_x3d, object &o_Ps, float x2d_threshold) {
		Np2D<float> x3ds(o_x3ds, -1, 3);
		const int num3ds = x3ds._rows;
		int32_t *labels_x3d = 0;
		if (!o_labels_x3d.is_none()) {
			Np1D<int32_t> tmp_labels_x3d(o_labels_x3d, num3ds);
			labels_x3d = tmp_labels_x3d._data;
		}
		Np3D<float> Ps(o_Ps, -1, 3, 4);
		const int numCams = Ps._rows;
	
		std::vector<int32_t> labels; labels.reserve(numCams*num3ds);
		std::vector<float> vels;
		float sc = _project_assign(num3ds, x3ds._data, labels_x3d, numCams, Ps._data, x2d_threshold, labels, &vels);
		object o_labels(newArrayFromVector<int32_t>(labels));
		object o_vels(newArray2DFromVector<float>(vels,2));
		return make_tuple(sc,o_labels,o_vels);
	}

	float _project_assign(const int num3ds, const float *x3ds, const int32_t *labels_x3d,
						  const int numCams, const float *Ps, const float x2d_threshold, 
						  std::vector<int32_t> &labels, std::vector<float> *p_vels = NULL)
	{

		std::vector<float> proj_x2ds; proj_x2ds.reserve(numCams*num3ds * 2);
		std::vector<int32_t> proj_labels; proj_labels.reserve(numCams*num3ds);
		std::vector<int32_t> proj_splits(numCams + 1);

		proj_splits[0] = 0;
		for (int ci = 0; ci < numCams; ++ci) {
			const float *P = Ps+ci*12;
			for (int xi = 0; xi < num3ds; ++xi) {
				project_x3d(proj_x2ds, proj_labels, x3ds + xi*3, P, labels_x3d?labels_x3d[xi]:xi);
			}
			proj_splits[ci + 1] = proj_labels.size();
		}
		int size = proj_labels.size();
		return _assign(proj_labels.size(), size?&proj_x2ds[0]:0, &proj_splits[0], size?&proj_labels[0]:0, x2d_threshold, labels, p_vels);
	}

	tuple project_assign_visibility(
		object &o_x3ds, object &o_labels_x3d, object &o_Ps, float x2d_threshold,
		boost::shared_ptr<ProjectVisibility> visibility
		) 
	{
		Np2D<float> x3ds(o_x3ds, -1, 3);
		const int num3ds = x3ds._rows;
		int32_t *labels_x3d = 0;
		if (!o_labels_x3d.is_none()) {
			Np1D<int32_t> tmp_labels_x3d(o_labels_x3d, num3ds);
			labels_x3d = tmp_labels_x3d._data;
		}
		Np3D<float> Ps(o_Ps, -1, 3, 4);
		const int numCams = Ps._rows;

		std::vector<int32_t> labels; labels.reserve(numCams*num3ds);
		std::vector<float> vels;
		float sc = _project_assign_visibility(
			num3ds, x3ds._data, labels_x3d, numCams, Ps._data, x2d_threshold, labels,
			visibility, &vels
			);
		object o_labels(newArrayFromVector<int32_t>(labels));
		object o_vels(newArray2DFromVector<float>(vels, 2));
		return make_tuple(sc, o_labels, o_vels);
	}

	float _project_assign_visibility(
		const int num3ds, const float *x3ds, const int32_t *labels_x3d,
		const int numCams, const float *Ps, const float x2d_threshold,
		std::vector<int32_t> &labels,
		boost::shared_ptr<ProjectVisibility> visibility,
		std::vector<float> *p_vels = NULL
		)
	{
		std::vector<float> proj_x2ds; proj_x2ds.reserve(numCams * num3ds * 2);
		std::vector<int32_t> proj_labels; proj_labels.reserve(numCams * num3ds);
		std::vector<int32_t> proj_splits(numCams + 1);

		proj_splits[0] = 0;
		for (int ci = 0; ci < numCams; ++ci)
		{
			const float *P = Ps + ci * 12;
			const float *x3d = x3ds;
			const float *normal_x3d = visibility ? visibility->normals() : NULL;
			const float *T = visibility && visibility->numTriangles() > 0 ? visibility->T(ci) : NULL;

			for (int xi = 0; xi < num3ds; ++xi, x3d += 3)
			{
				if (normal_x3d) normal_x3d += 3;

				if (visibility && visibility->numTriangles() > 0)
				{
					project_x3d_visibility(
						proj_x2ds, proj_labels, x3d, P, labels_x3d ? labels_x3d[xi] : xi, normal_x3d, visibility->numTriangles(), visibility->triangles(),
						T, visibility->triangleNormals(), visibility->intersectionThreshold(), visibility->generateNormals(), ci, 1
						);
				}
				else
				{
					project_x3d(proj_x2ds, proj_labels, x3d, P, labels_x3d ? labels_x3d[xi] : xi, normal_x3d);
				}
			}

			proj_splits[ci + 1] = proj_labels.size();
		}
		int size = proj_labels.size();
		return _assign(proj_labels.size(), size?&proj_x2ds[0]:0, &proj_splits[0], size?&proj_labels[0]:0, x2d_threshold, labels, p_vels);
	}
};

class HashCloud3D
{
	struct HI {
		int32_t hash;
		int32_t index;
		bool operator<(const HI &that) const { return ((hash < that.hash) || (hash == that.hash && index < that.index)); }
	};

	Np2D<float> _x3ds;
	int32_t     _size;
	float       _threshold;
	HI         *_hi;
	int32_t     _splits[0x1001];
public:
	/*Convert 3D points into a hashcloud via a threshold.
	Return values: hs are the the hash values of the original points. horder gives the index of the 3D points,
	in order of hash value. h_splits gives the range of points in horder that share each hash value.*/
	HashCloud3D(object &o_x3ds, float threshold) : _x3ds(o_x3ds,-1,3), _size(_x3ds._rows), _threshold(threshold)
	{
		_hi = new HI[_size];
		hash_12(_x3ds._data, _size, 0, reinterpret_cast<int32_t*>(_hi));
		std::sort(_hi, _hi+_size);
		for (int i = 0, j = 0; i < 0x1001; ++i) {
			while (j < _size && _hi[j].hash < i) ++j;
			_splits[i] = j;
		}
	}

	virtual ~HashCloud3D()
	{
		delete[] _hi;
	}

	/*A 12-bit hashing function. Each point is put at the centre of a unit cube, rounded down (floor),
	and converted to an integer by scaling, adding and modulus.
	Then, for any cube that includes the centre point, the hash of one of its corners will match.
	Suitable for hundreds points (expect hash collisions).*/
	void hash_12(const float *x3ds, const int32_t N, const float offset, int32_t *hi)
	{
		const float sc = 0.5/_threshold;
		for (int i = 0; i < N; ++i) {
			hi[2*i] = (int32_t(floor((x3ds[3*i]-offset)*sc))
				+0x10* int32_t(floor((x3ds[3*i+1]-offset)*sc))
				+0x100*int32_t(floor((x3ds[3*i+2]-offset)*sc))
				)&0xfff;
			hi[2*i+1] = i;
		}
	}

	/* Given 3D points, find all the possible matches in the hashcloud.
	* Consider each point as the centre of a unit cube; test the hashes of the eight corners to find
	* all candidates (-threshold<dx,dy,dz<threshold).
	* Might include some hash collisions. The return value is equivalent to a scipy.csr_matrix.*/
	tuple score(object &o_x3ds)
	{
		Np2D<float> x3ds(o_x3ds,-1,3);
		const int32_t size = x3ds._rows;
		std::vector<float> scores;
		std::vector<int32_t> matches;
		std::vector<int32_t> matches_splits;
		_score(size, x3ds._data, scores, matches, matches_splits);
		object o_scores(newArrayFromVector<float>(scores));
		object o_matches(newArrayFromVector<int32_t>(matches));
		object o_matches_splits(newArrayFromVector<int32_t>(matches_splits));
		return make_tuple(o_scores, o_matches, o_matches_splits);
	}

	void _score(const int32_t size, float *x3ds, std::vector<float> &scores, std::vector<int32_t> &matches, std::vector<int32_t> &matches_splits) {
		HI *hi = new HI[size];
		hash_12(x3ds, size, _threshold, reinterpret_cast<int32_t*>(hi));
		const int32_t offs[8] = { 0, 0x1, 0x10, 0x11, 0x100, 0x101, 0x110, 0x111 };
		matches_splits.resize(size+1);
		matches_splits[0] = 0;
		const float sqthresh = _threshold*_threshold;
		for (int32_t xi = 0; xi < size; ++xi) {
			HI &h = hi[xi];
			const float *src = x3ds+xi*3;
			const int32_t base = h.hash;
			for (int32_t oi = 0; oi < 8; ++oi) {
				const int32_t hash = (base+offs[oi]) & 0xfff;
				const int32_t r0 = _splits[hash], r1 = _splits[hash+1];
				for (int32_t ri = r0; ri < r1; ++ri) {
					const int32_t match = _hi[ri].index;
					const float *tgt = _x3ds[match];
					const float score = (src[0] - tgt[0])*(src[0] - tgt[0]) + (src[1] - tgt[1])*(src[1] - tgt[1]) + (src[2] - tgt[2])*(src[2] - tgt[2]);
					if (score < sqthresh) {
						matches.push_back(match);
						scores.push_back(score);
					}
				}
			}
			matches_splits[xi+1] = int32_t(matches.size());
		}
		delete[] hi;
	}
};

tuple intersect_rays_base(
	object &o_x2ds, object &o_splits, object &o_Ps, object &o_Ks, object &o_RTs, object &o_Ts, object &o_seed_x3ds, 
	float tilt_threshold = 0.0002, float x2d_threshold = 0.01, float x3d_threshold = 30.0, int min_rays = 3,
	int numPolishIts = 3, bool forceRayAgreement = false,
	boost::shared_ptr<ProjectVisibility> visibility = boost::shared_ptr<ProjectVisibility>()
	)
{
	/*
	Given 2D detections, we would like to find bundles of rays from different cameras that have a common solution.
	For each pair of rays, we can solve for a 3D point. Each such solve has a residual: we want to find low residual pairs.

	Closer together camera pairs and cameras with more unlabelled markers should have more matches.
	Visit the camera pairs by order of distance-per-unlabelled-marker score (lower is better).

	For a given camera pair, each ray can be given an order which is the tilt (angle between the ray from the camera to
	that ray and a line perpendicular to a reference plain containing both camera centres).
	
	tilt = asin(norm(raydir^(c2-c1)).ocdir))
	TODO: compare atan2(raydir^(c2-c1).ocdir,|raydir^(c2-c1)^ocdir|)
	
	Precisely the rays with the same tilt (within tolerance) intersect.
	This fails only if the first camera is looking directly at the second.

	For each pair of cameras, sort the unassigned rays by tilt and read off the matches.
	(DON'T match if there are two candidates with the same tilt on the same camera.)
	For each match, solve the 3D point.
	Naively, this costs ~NumDetections^2.
	However, if we project the point in all the cameras and assign rays then we can soak up all the rays in the other cameras.
	The maximum number of matches should be ~NumPoints.
	So the dominant cost becomes project assign (NumPoints * NumCameras using hash).

	Polish all the 3D points.
	Check for any 3D merges (DON'T merge if there are two rays from the same camera).
	Project all the points in all the cameras and reassign.
	Cull any points with fewer than 2 rays.
	Potentially repeat for the remaining unassigned rays.
	*/
	object o_E = compute_E(o_x2ds, o_splits, o_Ps);
	Np3D<float> E(o_E, -1, 2, 4);
	const int32_t num2ds = E._rows;
	const Np1D<int32_t> splits(o_splits, -1);
	const int numCameras = splits._size-1;
	Np3D<float> Ps(o_Ps, numCameras, 3, 4);
	const Np3D<float> Ks(o_Ks, numCameras, 3, 3);
	const Np3D<float> RTs(o_RTs, numCameras, 3, 4);
	const int32_t numDets = splits[numCameras];
	assert(numDets == num2ds);
	std::vector<int32_t> labels(numDets);
	for (int i = 0; i < numDets; ++i) labels[i] = -1;
	object o_rays = dets_to_rays(o_x2ds, o_splits, o_Ks, o_RTs);
	Np2D<float> rays(o_rays, numDets, 3);
	Np2D<float> Ts(o_Ts,numCameras,3);
	std::vector<float> tilt_axes(3*numCameras);
	for (int ci = 0; ci < numCameras; ++ci) 
	{
		const float *K = Ks[ci];
		const float K0 = -K[2], K1 = -K[5], K2 = K[0];
		const float *RT = RTs[ci];
		float *ta = &tilt_axes[3*ci];
		const float r0 = K0 * RT[0] + K1 * RT[4] + K2 * RT[8];
		const float r1 = K0 * RT[1] + K1 * RT[5] + K2 * RT[9];
		const float r2 = K0 * RT[2] + K1 * RT[6] + K2 * RT[10];
		float sc = r0*r0+r1*r1+r2*r2;
		if (sc != 0) sc = sqrt(1.0/sc);
		ta[0] = r0*sc;
		ta[1] = r1*sc;
		ta[2] = r2*sc;
	}
	HashCloud2DList clouds(o_x2ds, o_splits, x2d_threshold);
	std::vector<float> x3ds_ret; x3ds_ret.reserve(512*3);
	int num_seed_x3ds = 0;
	if (!o_seed_x3ds.is_none()) 
	{
		const Np2D<float> tmp_seed_x3ds(o_seed_x3ds, -1, 3);
		const float *sp = tmp_seed_x3ds._data;
		num_seed_x3ds = tmp_seed_x3ds._rows;
		for (int i = num_seed_x3ds*3; i; --i) x3ds_ret.push_back(*sp++);
		// initialise labels from seed_x3ds
		clouds._project_assign_visibility(num_seed_x3ds, tmp_seed_x3ds._data, NULL, numCameras, Ps._data, x2d_threshold, labels, visibility);
	}
	for (int ci = 0; ci < numCameras; ++ci) 
	{
		for (int cj = ci+1; cj < numCameras; ++cj) 
		{
			//def norm(a): return a / (np.sum(a**2)**0.5);
			std::vector<int32_t> ui,uj;
			for (int si = splits[ci]; si < splits[ci+1]; ++si)
				if (labels[si] == -1) ui.push_back(si);
			for (int sj = splits[cj]; sj < splits[cj+1]; ++sj)
				if (labels[sj] == -1) uj.push_back(sj);
			const int32_t len_ui = ui.size(), len_uj = uj.size();
			if (len_ui == 0 || len_uj == 0) continue;
			const float *Ts_ci = Ts[ci], *Ts_cj = Ts[cj], *ta_ci = &tilt_axes[3*ci];
			const float ax = Ts_cj[0] - Ts_ci[0];
			const float ay = Ts_cj[1] - Ts_ci[1];
			const float az = Ts_cj[2] - Ts_ci[2];
			const float tx = ta_ci[0], ty = ta_ci[1], tz = ta_ci[2];
			std::vector<float> tilt_i, tilt_j;
			for (int ti = 0; ti < len_ui; ++ti) 
			{
				const float *ri = rays[ui[ti]];
				const float rx = ri[0], ry = ri[1], rz = ri[2];
				const float cx = ry*az-rz*ay, cy = rz*ax-rx*az, cz = rx*ay-ry*ax;
				const float sc = pow(cx*cx+cy*cy+cz*cz, -0.5f);
				const float val = (cx*tx + cy*ty + cz*tz)*sc;
				tilt_i.push_back(val);
			}
			for (int tj = 0; tj < len_uj; ++tj) 
			{
				const float *rj = rays[uj[tj]];
				const float rx = rj[0], ry = rj[1], rz = rj[2];
				const float cx = ry*az-rz*ay, cy = rz*ax-rx*az, cz = rx*ay-ry*ax;
				const float sc = pow(cx*cx+cy*cy+cz*cz, -0.5f);
				const float val = (cx*tx + cy*ty + cz*tz)*sc;
				tilt_j.push_back(val);
			}
			std::vector<int32_t> io,jo;
			argsort(tilt_i.begin(), tilt_i.end(), io);
			argsort(tilt_j.begin(), tilt_j.end(), jo);
			std::vector<int32_t> data;
			const int32_t len_io = io.size(), len_jo = jo.size();
			for (int ii = 0,ji = 0; ii < len_io && ji < len_jo;) {
				const float d0 = tilt_i[io[ii]], d1 = tilt_j[jo[ji]];
				const float diff = d0 - d1;
				if (abs(diff) < tilt_threshold) {
					// test for colliding pairs
					if (ii+1 < len_io && tilt_i[io[ii+1]]-d0 < tilt_threshold) { ii+=2; continue; }
					if (ji+1 < len_jo && tilt_j[jo[ji+1]]-d1 < tilt_threshold) { ji+=2; continue; }
					// test for colliding triples
					if (ii > 0 && d0-tilt_i[io[ii-1]] < tilt_threshold) { ii++; continue; }
					if (ji > 0 && d1-tilt_j[jo[ji-1]] < tilt_threshold) { ji++; continue; }
					data.push_back(ui[io[ii]]);
					data.push_back(uj[jo[ji]]);
					ii++;
					ji++;
				}
				else if (diff < 0) ii++;
				else               ji++;
			}
			// intersect rays
			for (int di = 0; di < data.size(); di+=2) 
			{
				float x3d[3];
				linsolveN3(E._data, &data[di], 2, x3d);
				std::vector<int32_t> tmp;
				float sc = 0;
				if (visibility)
					sc = clouds._project_assign_visibility(1, x3d, NULL, numCameras, Ps._data, x2d_threshold, tmp, visibility);
				else
					sc = clouds._project_assign(1, x3d, NULL, numCameras, Ps._data, x2d_threshold, tmp);

				int j = 0;
				for (int i = 0; i < tmp.size(); ++i) { if (tmp[i] == 0) tmp[j++] = i; }
				if (j >= min_rays) 
				{
					int k = 0;
					for (int i = 0; i < j; ++i) { if (labels[tmp[i]] == -1) tmp[k++] = tmp[i]; }
					if (k >= min_rays) 
					{
						const int32_t index = x3ds_ret.size()/3;
						for (int i = 0; i < k; ++i) { labels[tmp[i]] = index; }
						x3ds_ret.push_back(x3d[0]);
						x3ds_ret.push_back(x3d[1]);
						x3ds_ret.push_back(x3d[2]);
					}
				}
			}
		}
	}
	// TODO polish, merge, reassign, cull, repeat
	
	if (false) // TODO merge
	{
		object o_x3ds(newArray2DFromVector<float>(x3ds_ret, 3));
		HashCloud3D cloud(o_x3ds, x3d_threshold);
		std::vector<float> scores;
		std::vector<int32_t> matches, matches_splits;
		cloud._score(x3ds_ret.size()/3, &x3ds_ret[0], scores, matches, matches_splits);
		for (int mi = 1; mi < matches_splits.size(); ++mi) {
			int li = mi-1;
			int i0 = matches_splits[li], i1 = matches_splits[li+1];
			if (i1 - i0 > 1) {
				std::vector<int32_t> collisions;
				for (int i = i0; i < i1; ++i) { if (scores[i] < x3d_threshold*x3d_threshold) collisions.push_back(i); }
				if (collisions.size() > 1) {
					//print 'merger',li,i0,i1,scores[i0:i1] # TODO merge these (frame 7854)
				}
			}
		}
	}

	// now cull the seed_x3ds, because they could confuse matters
	if (num_seed_x3ds) 
	{
		for (int li = 0; li < labels.size(); ++li) { if (labels[li] < num_seed_x3ds) labels[li] = -1; }
	}

	// Final polish
	for (int c = 0; c < numPolishIts-1; ++c)
	{
		if (labels.size()) solve_x3ds_only(o_E, &labels[0], false, x3ds_ret, min_rays, rays, forceRayAgreement);
		// throw away the single rays and their 3d points by renumbering the generated 3d points
		if (x3ds_ret.size())
		{
			clouds._project_assign_visibility(x3ds_ret.size() / 3, &x3ds_ret[0], NULL, numCameras, Ps._data, x2d_threshold, labels, visibility);
		}
		else
		{
			std::fill(labels.begin(), labels.end(), -1);
		}
	}

	return solve_x3ds_base2(E._data, labels.size(), labels.size()?&labels[0]:NULL, true, min_rays);
}

tuple intersect_rays(
	object &o_x2ds, object &o_splits, object &o_Ps, object &o_Ks, object &o_RTs, object &o_Ts, object &o_seed_x3ds,
	float tilt_threshold = 0.0002, float x2d_threshold = 0.01, float x3d_threshold = 30.0, int min_rays = 3
	)
{
	return intersect_rays_base(o_x2ds, o_splits, o_Ps, o_Ks, o_RTs, o_Ts, o_seed_x3ds, tilt_threshold, x2d_threshold, x3d_threshold, min_rays);
}

tuple intersect_rays2(
	object &o_x2ds, object &o_splits, object &o_Ps, object &o_Ks, object &o_RTs, object &o_Ts, object &o_seed_x3ds,
	float tilt_threshold = 0.0002, float x2d_threshold = 0.01, float x3d_threshold = 30.0, int min_rays = 3,
	int numPolishIts = 3, bool forceRayAgreement = false
	)
{
	return intersect_rays_base(
		o_x2ds, o_splits, o_Ps, o_Ks, o_RTs, o_Ts, o_seed_x3ds, tilt_threshold, 
		x2d_threshold, x3d_threshold, min_rays, numPolishIts, forceRayAgreement);
}

// [[160, 0, 0],[0, 0, 0],[-80, 0, 0],[0, 0, -120],[0, 0, -240]]
//  80  160
// C--B----A
//    |.120
//    D
//    |.120
//    E
// default float values: 2.0, 0.5, 0.01, 0.07
void label_T_wand(object &o_x2ds_data, object &o_x2ds_splits, object &o_x2ds_labels,
				  float ratio, float x2d_threshold, float straightness_threshold, float match_threshold)
{
	// A T-wand is defined to be a 5-point wand where three points are unevenly spaced on a line according to the given ratio;
	// and the central of these three points is at one end of a perpendicular line with the other two, which are evenly spaced.
	const Np2D<float>   x2ds_data(o_x2ds_data, -1, 2);
	const int           num2ds = x2ds_data._rows;
	const Np1D<int32_t> x2ds_splits(o_x2ds_splits, -1);
	const int           numCams = x2ds_splits._size-1;
	Np1D<int32_t>       x2ds_labels(o_x2ds_labels, num2ds); // the return value

	for (int di = 0; di < num2ds; ++di) x2ds_labels[di] = -1;

	std::vector<int32_t> order(5);
	std::vector<int32_t> straightOrder(3);
	float tmp[3];
	float xs[10];
	for (int ci = 0; ci < numCams; ++ci) {
		const int c0 = x2ds_splits[ci], c1 = x2ds_splits[ci+1];
		const int numDets = c1-c0; // number of detections this camera
		if (numDets < 5) continue; // can't hope to label less than 5 points!
		const float *x2ds = x2ds_data[c0]; // x2ds[2*di + ei] = di^th detection, ei^th channel (0=x,1=y)
		int32_t *labels = &x2ds_labels[c0];
		HashCloud2D cloud(x2ds, numDets, x2d_threshold);
		std::vector<float> scores;
		std::vector<int32_t> matches;
		std::vector<int32_t> matches_splits; matches_splits.resize(1+numDets);
		cloud._score_nearest_N(5, x2ds, numDets, scores, matches, &matches_splits[0]);
		//cloud._score(x2ds, numDets, scores, matches, &matches_splits[0]);
		for (int si = 0; si < scores.size(); ++si) scores[si] = sqrtf(scores[si]); // sqrts are more useful here
		for (int xi = 0; xi < numDets; ++xi) {
			const int32_t m0 = matches_splits[xi], m1 = matches_splits[xi+1];
			if (m1 < m0+5) continue; // must have 5 neighbours (including itself)
			const float *su = &scores[m0];
			const int32_t *mu = &matches[m0];
			argsort(su,su+m1-m0, order);
			assert(mu[order[0]] == xi); // we assume xi is the central point and ms are the other four points
			const float xix = x2ds[2*xi+0], xiy = x2ds[2*xi+1];
			for (int i = 0; i < 5; ++i) {
				xs[2*i+0] = x2ds[2*mu[order[i]]+0] - xix;
				xs[2*i+1] = x2ds[2*mu[order[i]]+1] - xiy;
			}
			// two of the points should be on a straight line with xi, having the correct ratio and opposite directions
			// the other two point should be on a straight line with xi, having ratio 2.0 and same direction
			// since BC<BA and BD<BE, the closest point is either C or D; and the furthest point is either E or A
			// find which point is most straight with the closest
			const float px = -xs[2*1+1]/su[order[1]], py = xs[2*1+0]/su[order[1]]; // perpendicular unit vector
			for (int i = 0; i < 3; ++i) tmp[i] = (xs[2*(2+i)]*px+xs[2*(2+i)+1]*py)/su[order[2+i]];
			for (int i = 0; i < 3; ++i) tmp[i] *= tmp[i];
			argsort(tmp, tmp+3, straightOrder);
			//std::cerr << tmp[straightOrder[0]] << ", " <<tmp[straightOrder[1]] << ", " << tmp[straightOrder[2]] << ", " << std::endl;
			if (tmp[straightOrder[0]] > straightness_threshold) continue; // not straight enough
			if (tmp[straightOrder[1]] < straightness_threshold*2) continue; // too straight (four points in a row)
			const int32_t xj = 2+straightOrder[0]; // pick the straightest
			const float sense = (xs[2*1]*xs[2*xj]+xs[2*1+1]*xs[2*xj+1])/(su[order[1]]*su[order[xj]]);
			const float rat = su[order[xj]]/su[order[1]];
			const int32_t xk = (xj==2?3:2), xl = (xj==4?3:4);
			const float sense2 = (xs[2*xk]*xs[2*xl]+xs[2*xk+1]*xs[2*xl+1])/(su[order[xk]]*su[order[xl]]);
			const float rat2 = su[order[xl]]/su[order[xk]];
			//std::cerr << "ci " << ci << " xi " << xi << " sense " << sense << " sense2 " << sense2 << " rat " << rat << " rat2 " << rat2 << std::endl;
			if (sense > 1.0 - straightness_threshold*2 && sense < 1.0 + straightness_threshold*2 &&
				sense2 > -1.0 - straightness_threshold*2 && sense2 < -1.0 + straightness_threshold*2 &&
				rat > 2.0*(1 - match_threshold) && rat < 2.0*(1 + match_threshold) &&
				rat2 > ratio*(1 - match_threshold) && rat2 < ratio*(1 + match_threshold))
			{
				labels[mu[order[xl]]] = 0;
				labels[mu[order[0]]] = 1;
				labels[mu[order[xk]]] = 2;
				labels[mu[order[1]]] = 3;
				labels[mu[order[xj]]] = 4;
				break;
			}
			else if (sense2 > 1.0 - straightness_threshold*2 && sense2 < 1.0 + straightness_threshold*2 &&
				sense > -1.0 - straightness_threshold*2 && sense < -1.0 + straightness_threshold*2 &&
				rat2 > 2.0*(1 - match_threshold) && rat2 < 2.0*(1 + match_threshold) &&
				rat > ratio*(1 - match_threshold) && rat < ratio*(1 + match_threshold))
			{
				labels[mu[order[xj]]] = 0;
				labels[mu[order[0]]] = 1;
				labels[mu[order[1]]] = 2;
				labels[mu[order[xk]]] = 3;
				labels[mu[order[xl]]] = 4;
				break;
			}
		}
	}
}

struct hypothesis {
	float   score;
	int32_t *g2x;
	inline bool operator<(const hypothesis &that) const { return score < that.score; }
};


float label_from_graph(object &o_x3ds, object &o_g2l, object &o_graphSplits, object &o_backlinks, object &o_DM,
					   const int32_t keepHypotheses, const float penalty, object &o_l2x)
{
	//A hypothesis is a g2x labelling: for each graph node, the assignment of which point.
	//Grow a list of hypotheses by adding each node of the graph. Update the hypotheses to include that node (for every assignment).
	//Keep the hypotheses sorted by score and limit the size and worst score.
	//Returns in l2x the labels for the best hypothesis and (by value) the score.
	const Np2D<float> x3ds(o_x3ds, -1, 3);
	const int numPoints = x3ds._rows;
	float *D = new float[numPoints*numPoints];
	const float *Xi = x3ds._data;
	float *Di = D;
	for (int xi = 0; xi < numPoints; ++xi, Xi+=3) {
		const float Xix = Xi[0], Xiy = Xi[1], Xiz = Xi[2];
		const float *Xj = x3ds._data;
		for (int xj = 0; xj < numPoints; ++xj, Xj+=3, ++Di) {
			const float dx = Xj[0] - Xix, dy = Xj[1] - Xiy, dz = Xj[2] - Xiz;
			*Di = sqrtf(dx*dx + dy*dy + dz*dz);
		}
	}
	Np1D<int32_t> l2x(o_l2x, -1);
	const int numLabels = l2x._size;
	Np1D<int32_t> g2l(o_g2l, numLabels);
	Np1D<int32_t> graphSplits(o_graphSplits, numLabels+1);
	Np1D<int32_t> backlinks(o_backlinks, graphSplits[numLabels]);
	Np2D<float> DM(o_DM, graphSplits[numLabels], 2);
	
	const int maxHypotheses = keepHypotheses+numPoints+1;
	int32_t *g2x = new int32_t[maxHypotheses * numLabels * 2]; // allocate some memory
	hypothesis* hypotheses_in = new hypothesis[maxHypotheses];
	hypothesis* hypotheses_out = new hypothesis[maxHypotheses];
	for (int hi = 0; hi < maxHypotheses; ++hi) {
		hypotheses_in [hi].g2x = g2x + hi * numLabels;
		hypotheses_out[hi].g2x = g2x + (maxHypotheses + hi) * numLabels;
	}
	// initial state
	hypotheses_in[0].score = 0;
	float bestScore = 0;
	int32_t hypotheses_in_size = 1;
	int32_t *in_h = new int32_t[numPoints+1] + 1; // in_h[-1] is always false, but it's needed
	for (int xi = -1; xi < numPoints; ++xi) in_h[xi] = 0;
	const int32_t *graph_splits = graphSplits._data;
	const int32_t *graph_backlinks = backlinks._data;
	const float penalty2 = penalty*2;

	for (int32_t gi = 0; gi < numLabels; ++gi) {
		const int32_t b0 = graph_splits[gi], b1 = graph_splits[gi+1]; //const int32_t *backEdges = graph_backlinks+b0
		int32_t hypotheses_out_size = 0;
		bestScore += penalty*(b1-b0); // len(backEdges) = (b1-b0)
		float thresholdScore = bestScore+penalty2;
		for (int hi = 0; hi < hypotheses_in_size; ++hi) {
			hypothesis &hin = hypotheses_in[hi];
			if (hin.score > thresholdScore) continue;
			for (int xi = 0; xi < numPoints; ++xi) in_h[xi] = 0;
			for (int g2 = 0; g2 < gi; ++g2) in_h[hin.g2x[g2]] = 1;
			{
				float sc = hin.score+penalty*(b1-b0);
				hypothesis &hout = hypotheses_out[hypotheses_out_size++];
				hout.score = sc;
				memcpy(hout.g2x, hin.g2x, sizeof(int32_t)*gi);
				hout.g2x[gi] = -1;
				if (sc < bestScore) bestScore = sc;
			}
			for (int xi = 0; xi < numPoints; ++xi) {
				if (in_h[xi]) continue; // already assigned
				float sc = hin.score;
				const float *D_xi = D+xi*numPoints;
				for (int b = b0; b < b1; ++b) {
					const int32_t xj = hin.g2x[graph_backlinks[b]];
					if (xj == -1) { sc += penalty; }
					else          { float ds = (D_xi[xj]-DM[b][0])*DM[b][1]; sc += ds*ds; }
				}
				if (sc > thresholdScore) continue;
				hypothesis &hout = hypotheses_out[hypotheses_out_size++];
				hout.score = sc;
				memcpy(hout.g2x, hin.g2x, sizeof(int32_t)*gi);
				hout.g2x[gi] = xi;
				if (sc < bestScore) bestScore = sc;
			}
			if (hypotheses_out_size > keepHypotheses) {
				std::nth_element(hypotheses_out, hypotheses_out+keepHypotheses, hypotheses_out+hypotheses_out_size);
				hypotheses_out_size = keepHypotheses;
				thresholdScore = std::min(bestScore+penalty2, hypotheses_out[keepHypotheses].score);
			}
		}
		std::sort(hypotheses_out, hypotheses_out+hypotheses_out_size);
		std::swap(hypotheses_in, hypotheses_out);
		hypotheses_in_size = std::min(keepHypotheses,hypotheses_out_size);
	}
	float ret = hypotheses_in[0].score;
	for (int gi = 0; gi < numLabels; ++gi) l2x[g2l[gi]] = hypotheses_in[0].g2x[gi];

	delete[] hypotheses_in;
	delete[] hypotheses_out;
	delete[] (in_h-1);
	delete[] g2x;
	delete[] D;
	return ret;
}

void dm_from_l3ds(object &o_l3ds, object &o_ws, object &o_M, object &o_W)
{
	// Given l3ds, a numFrames x numLabels x 3 data matrix of animating labelled 3d points,
	// and ws a numFrames x numLabels weights matrix, compute the distance matrix.
	const Np3D<float> l3ds(o_l3ds, -1, -1, 3);
	int numFrames = l3ds._rows;
	int numLabels = l3ds._cols;
	const Np2D<float> ws(o_ws, numFrames, numLabels);
	Np2D<float> M(o_M,numLabels,numLabels);
	Np2D<float> W(o_W,numLabels,numLabels);
	for (int li = 0; li < numLabels; ++li) {
		for (int lj = 0; lj < li; ++lj) {
			double sw = 0;
			double d2_sum = 0;
			double d4_sum = 0;
			for (int fi = 0; fi < numFrames; ++fi) {
				const float *l3ds_fi = l3ds[fi];
				const float w = ws[fi][li] * ws[fi][lj];
				if (w == 0.0) continue;
				const float dx = (l3ds_fi[3*li+0]-l3ds_fi[3*lj+0]);
				const float dy = (l3ds_fi[3*li+1]-l3ds_fi[3*lj+1]);
				const float dz = (l3ds_fi[3*li+2]-l3ds_fi[3*lj+2]);
				const double d2 = sqrtf(dx*dx+dy*dy+dz*dz)*w;
				sw += w;
				d2_sum += d2;
				d4_sum += d2*d2;
			}
			sw = (sw > 0 ? 1.0/sw : 0);
			double d2_mean = d2_sum*sw;
// 			d4_sum = 0;
// 			for (int fi = 0; fi < numFrames; ++fi) {
// 				const float *l3ds_fi = l3ds[fi];
// 				const float w = ws[fi][li] * ws[fi][lj];
// 				if (w == 0.0) continue;
// 				const float dx = (l3ds_fi[3*li+0]-l3ds_fi[3*lj+0]);
// 				const float dy = (l3ds_fi[3*li+1]-l3ds_fi[3*lj+1]);
// 				const float dz = (l3ds_fi[3*li+2]-l3ds_fi[3*lj+2]);
// 				const double d2 = sqrtf(dx*dx+dy*dy+dz*dz)*w;
// 				d4_sum += (d2 - d2_mean)*(d2 - d2_mean);
// 			}
			//double d2_var = d4_sum*sw;
			double d2_var = d4_sum*sw - d2_mean*d2_mean;
			d2_var = sqrtf(1.0/(d2_var+1.0));
			if (d2_mean < 20 || d2_mean > 1000) d2_var = 0;
			M[li][lj] = M[lj][li] = d2_mean;
			W[li][lj] = W[lj][li] = d2_var;
		}
	}
}

// Time for the X2D reader to be moved to C

str unpack_str(const char *&offset,const char *eos)
{
	const int size = (*(const int32_t *)offset)-4; offset += 4;
	//std::cerr << std::dec << size << std::endl;
	//for (int i = 0; i < size; ++i) std::cerr << offset[i] << std::endl;
	str ret(offset, size);
	offset += size;
	return ret;
}

// TODO turn this into a numpy array
tuple unpack_list_Hii(const int t, const char *&offset, const char *eos)
{
	const int tmp = *(const uint16_t *)offset; offset += 2;
	assert(t == tmp);
	list ret;
	int size = *(const int32_t *)offset; offset += 4;
	while (offset < eos && size--) {
		//std::cerr << size << std::endl;
		const int val1 = *(const uint16_t *)offset; offset += 2;
		const int val2 = *(const int32_t *)offset; offset += 4;
		const int val3 = *(const int32_t *)offset; offset += 4;
		ret.append(make_tuple(val1,val2,val3));
	}
	return make_tuple(t,ret);
}

// TODO turn this into a numpy array
tuple unpack_list_HiiHH(const int t, const char *&offset, const char *eos)
{
	const int tmp = *(const uint16_t *)offset; offset += 2;
	assert(t == tmp);
	list ret;
	int size = *(const int32_t *)offset; offset += 4;
	while (offset < eos && size--) {
		const int val1 = *(const uint16_t *)offset; offset += 2;
		const int val2 = *(const int32_t *)offset; offset += 4;
		const int val3 = *(const int32_t *)offset; offset += 4;
		const int val4 = *(const uint16_t *)offset; offset += 2;
		const int val5 = *(const uint16_t *)offset; offset += 2;
		ret.append(make_tuple(val1,val2,val3,val4,val5));
	}
	return make_tuple(t,ret);
}

list unpack_camera_info_list(const char *&offset, const char *eos, std::vector<int> &cam_shapes)
{
	list ret;
	int size = *(const int32_t *)offset; offset += 4;
	while (offset < eos && size--) {
		const int val4_1 = *(const int32_t *)offset; offset += 4;
		const int val4_2 = *(const int32_t *)offset; offset += 4;
		const int val4_3 = *(const int32_t *)offset; offset += 4;
		const int val4_4 = *(const int32_t *)offset; offset += 4;
		const double val4_5 = *(const double *)offset; offset += 8;
		str val4_6 = unpack_str(offset,eos);
		const int val4_7 = *(const int32_t *)offset; offset += 4;
		str val4_8 = unpack_str(offset,eos);
		const double val4_9 = *(const double *)offset; offset += 8;
		str val4_10 = unpack_str(offset,eos);
		tuple val4 = make_tuple(val4_1,val4_2,val4_3,val4_4,val4_5,val4_6,val4_7,val4_8,val4_9,val4_10);
		ret.append(val4);
		cam_shapes.push_back(val4_1);
		cam_shapes.push_back(val4_2);
	}
	return ret;
}

list unpack_camera_data_list(const char *&offset, const char *eos)
{
	list ret;
	while (offset < eos) {
		const int t = *(const uint16_t *)offset;
		//std::cerr << std::hex << t << std::endl;
		if (t == 0xdfaa) {
			ret.append(unpack_list_HiiHH(t,offset,eos));// ['HiiHH'], # CentroidData {CameraId/Frame/RowCount/?/?}
		}
		else {
// 			case 0xdfd0: // CentroidTrackData {CameraId/Frame/RowCount}
// 			case 0xdfbb: // GreyScaleData {CameraId/Frame/RowCount}
// 			case 0xdfee: // ThresholdData {CameraId/Frame/RowCount}
// 			case 0xdfcc: // EdgeData {CameraId/Frame/RowCount}
			ret.append(unpack_list_Hii(t,offset,eos));
		}
	}
	return ret;
}

list unpack_x2d_header(const char *&offset, const char *eos, std::vector<int> &cam_shapes)
{
	list ret;
	while(offset < eos) {
		const int t = *(const uint16_t *)offset; offset += 2;
		const int size = *(const int32_t *)offset; offset += 4;
		const char *eob = offset + size;
		//std::cerr << t << "," << size << std::endl;
		switch (t) {
			case 0x00cc: // NumCameras
			case 0x00fc: // NumFrames
			case 0x0dfc: // FrameRate
			{
				assert(size == 4);
				const int val1 = *(const int32_t *)offset; offset += 4;
				ret.append(make_tuple(t,val1));
				break;
			}
			case 0x0010: // IndexIndex
			case 0x00d0: // FrameIndex
			{
				assert(size == 8);
				const int64_t val1 = *(const int64_t *)offset; offset += 8;
				ret.append(make_tuple(t,val1));
				break;
			}
			case 0xfddf: // 'Hi', fddf, 0 ### Is this a special byte marker??
			{
				assert(size == 6);
				const int val1 = *(const uint16_t *)offset; offset += 2;
				const int val2 = *(const int32_t *)offset; offset += 4;
				ret.append(make_tuple(t,val1,val2));
				break;
			}
			case 0x00df: // list of camera data
			{
				ret.append(make_tuple(t,unpack_camera_data_list(offset,eob)));
				break;
			}
			case 0x0000: // (0xcfab, 'di', 0xcfaa, ['iiiidsisds']), # CameraInfo {captureRate,unknown} (12bys),{ImageWidth,ImageHeight,CameraID,UserID,PixelAspectRatio,CameraType,0,'0 ',CircularityThreshold,DisplayType}
			{
				tuple val1 = unpack_list_Hii(0xcfab, offset, eob);
				const double val2_1 = *(const double *)offset; offset += 8;
				const int val2_4 = *(const int32_t *)offset; offset += 4;
				tuple val2 = make_tuple(val2_1,val2_4);
				tuple val3 = unpack_list_Hii(0xcfaa, offset, eob);
				list val4 = unpack_camera_info_list(offset, eob, cam_shapes);
				ret.append(make_tuple(t, val1, val2, val3, val4));
				break;
			}
			case 0x0001: //'HiHiis', #TimeCode 'None/Film/PAL/NTSC/NTSC-Drop'
			{
				const int val1 = *(const uint16_t *)offset; offset += 2;
				const int val2 = *(const int32_t *)offset; offset += 4;
				const int val3 = *(const uint16_t *)offset; offset += 2;
				const int val4 = *(const int32_t *)offset; offset += 4;
				const int val5 = *(const int32_t *)offset; offset += 4;
				str val6 = unpack_str(offset,eob);
				ret.append(make_tuple(t,val1,val2,val3,val4,val5,val6));
				break;
			}
			case 0xffff: // finished
			{
				offset -= 6;
				return ret;
			}
			default:
			{
				std::cerr << "oops: discarding " << t << std::endl;
				offset += size;
			}
		}
	}
	return ret;
}

int unpack_camera(const char *&offset, const char *eos, const std::vector<int> &cam_shapes,
				  std::vector<float> &centroids, std::vector<int32_t> &centroids_splits,
				  std::vector<int32_t> &tracks, std::vector<int32_t> &tracks_splits,
				  std::vector<unsigned char> &greyscales, std::vector<int32_t> &greyscales_splits,
				  std::vector<unsigned char> &thresholds, std::vector<int32_t> &thresholds_splits,
				  std::vector<uint16_t> &edges, std::vector<int32_t> &edges_splits)
{
	//const int id = *(const int32_t *)offset; offset += 4;
	//assert(id == centroids_splits.size()-1);
	const int id = centroids_splits.size()-1; offset += 4;
	const int cw = cam_shapes[id*2], ch = cam_shapes[id*2+1];
	int mw = cw, mh = ch;
	for (int i = 0; i < cam_shapes.size(); i += 2) {
		mw = std::max(mw,cam_shapes[i]);
		mh = std::max(mh,cam_shapes[i+1]);
	}
	const int cid = *(const uint16_t *)offset; offset += 2;
	//std::cerr << "cid " << cid << std::endl;
	assert(cid == 0xdfaa);
	int bsize = *(const int32_t *)offset; offset += 4; // TODO use bsize
	const int centroids_size = *(const int32_t *)offset; offset += 4;
	//std::cerr << "centroids_size " << centroids_size << std::endl;
	const int32_t cs = centroids.size();
	centroids.resize(cs + centroids_size*4);
	centroids_splits.push_back(centroids.size()/4);
	// convert the centroids into [-1,1]x[-aspect,aspect] coordinates
	// this is tricky because they are scaled so that the LARGEST pixel values fit
	// we also flip the y-axis here
	const float norm = (1.0/(256*256*256))/float(cw);
	const float scalex = 2.0*mw*norm;
	const float scaley = -2.0*mh*norm;
	const float aspect = float(ch)/cw;
	const float scalec = 1.0/(256*256);
	if (centroids_size) {
		float *cp = &centroids[cs];
		for (int i = 0; i < centroids_size; ++i) {
			*cp++ = ((*(const int32_t *)offset) & 0xffffff)*scalex - 1.0; offset += 3;
			*cp++ = ((*(const int32_t *)offset) & 0xffffff)*scaley + aspect; offset += 3;
			*cp++ = ((*(const int32_t *)offset) & 0xffffff)*scalex; offset += 3;
			*cp++ = ((*(const uint16_t *)offset)*scalec); offset += 2;
			//std::cerr << std::dec << i << ":" << cd[0] << " " << cd[1] << " " << cd[2] << " " << cd[3] << std::endl;
		}
	}

	const int cdid = *(const uint16_t *)offset; offset += 2;
	//std::cerr << std::hex << cdid << std::endl;
	assert(cdid == 0xdfd0 || cdid == 0xdfdd); // TODO look into this! how is the dfdd different from the dfd0? found in M:\ViconDB\MLF\VW_Passat
	bsize = *(const int32_t *)offset; offset += 4;
	const int centroid_tracks_size = *(const int32_t *)offset; offset += 4;
	//std::cerr << "centroid_tracks_size " << centroid_tracks_size <<std::endl;
	const int32_t ts = tracks.size();
	tracks.resize(ts + centroid_tracks_size*4);
	tracks_splits.push_back(tracks.size()/4);
	if (centroid_tracks_size) memcpy(&tracks[ts], offset, centroid_tracks_size*16); offset += centroid_tracks_size*16;

	const int gsid = *(const uint16_t *)offset; offset += 2;
	//std::cerr << std::hex << gsid << std::endl;
	assert(gsid == 0xdfbb);
	const int greyscales_size = *(const int32_t *)offset; offset += 4;
	//std::cerr << "greyscales_size " << greyscales_size <<std::endl;
	const int32_t gs = greyscales.size();
	greyscales.resize(gs + greyscales_size);
	greyscales_splits.push_back(greyscales.size());
	if (greyscales_size) memcpy(&greyscales[gs], offset, greyscales_size); offset += greyscales_size; // TODO decode this list of grayscales...

	offset += 4; // this just seems to be a bug in the file format

	const int thid = *(const uint16_t *)offset; offset += 2;
	//std::cerr << std::hex << thid << std::endl;
	assert(thid == 0xdfee);
	bsize = *(const int32_t *)offset; offset += 4;
	const int thresholds_size = *(const int32_t *)offset; offset += 4;
	const int32_t ths = thresholds.size();
	thresholds.resize(ths + thresholds_size);
	thresholds_splits.push_back(thresholds.size());
	if (thresholds_size) memcpy(&thresholds[ths], offset, thresholds_size); offset += thresholds_size;

	const int eid = *(const uint16_t *)offset; offset += 2;
	assert(eid == 0xdfcc);
	bsize = *(const int32_t *)offset; offset += 4;
	const int edges_size = *(const int32_t *)offset; offset += 4;
	const int32_t es = edges.size();
	edges.resize(es + edges_size*3);
	edges_splits.push_back(edges.size()/3);
	if (edges_size) memcpy(&edges[es], offset, edges_size*6); offset += edges_size*6;

	return id;
}


list unpack_cameras_list(const char *&offset, const char *eos, const std::vector<int> &cam_shapes) {
	std::vector<float> centroids; std::vector<int32_t> centroids_splits(1);
	std::vector<int32_t> tracks; std::vector<int32_t> tracks_splits(1);
	std::vector<unsigned char> greyscales; std::vector<int32_t> greyscales_splits(1);
	std::vector<unsigned char> thresholds; std::vector<int32_t> thresholds_splits(1);
	std::vector<uint16_t> edges; std::vector<int32_t> edges_splits(1);
	while(offset < eos) {
		const int tmp = *(const uint16_t *)offset; offset += 2;
		//std::cerr << std::hex << tmp << std::endl;
		assert(tmp == 0xcccc);
		int size = *(const int32_t *)offset; offset += 4;
		//std::cerr << "unpack_cameras_list " << size << std::endl;
		unpack_camera(offset, offset+size, cam_shapes, centroids, centroids_splits, tracks, tracks_splits, greyscales, greyscales_splits, thresholds, thresholds_splits, edges, edges_splits);
	}
	list ret;
	ret.append(newArray2DFromVector<float>(centroids, 4));
	ret.append(newArrayFromVector<int32_t>(centroids_splits));
	ret.append(newArray2DFromVector<int32_t>(tracks, 4));
	ret.append(newArrayFromVector<int32_t>(tracks_splits));
	ret.append(newArrayFromVector<unsigned char>(greyscales));
	ret.append(newArrayFromVector<int32_t>(greyscales_splits));
	ret.append(newArrayFromVector<unsigned char>(thresholds));
	ret.append(newArrayFromVector<int32_t>(thresholds_splits));
	ret.append(newArray2DFromVector<uint16_t>(edges, 3));
	ret.append(newArrayFromVector<int32_t>(edges_splits));
	return ret;
}

list unpack_frames(const char *&offset, const char *eos, const std::vector<int> &cam_shapes) {
	list ret;
	while(offset < eos) {
		const int tmp = *(const uint16_t *)offset; offset += 2;
		if (tmp != 0xffff) { offset -= 2; break; }
		const int size = *(const int32_t *)offset; offset += 4;
		assert(offset+size <= eos);
		const int fi = *(const int32_t *)offset; offset += 4;
		//std::cerr << std::dec << fi << std::endl;
		ret.append(unpack_cameras_list(offset, offset+size, cam_shapes));
	}
	return ret;
}

#if PY_MAJOR_VERSION >= 3
#define PyString_Size PyBytes_Size
#define NUMPY_IMPORT_ARRAY_RETVAL
#endif

dict decode_X2D(str s) {
	const char *offset = extract<char const*>(s);
	const char *eos = offset + PyString_Size(s.ptr()); // bugfix: python's __len__() method returns a 32 bit int as opposed to len() which returns a long
	const int format = *(const int32_t *)offset; offset += 4;
	assert(format == 3);
	std::vector<int> cam_shapes;
	list header = unpack_x2d_header(offset, eos, cam_shapes);
	list frames = unpack_frames(offset, eos, cam_shapes);
	//list index = unpack_index(offset, eos);
	dict ret;
	ret["header"] = header;
	ret["frames"] = frames;
	return ret; // TODO , index);
// 	#print 'index', offset
// 	count = IO.unpack_from('<i',s,offset)[0]; offset += 4
// 	for f in xrange(count):
// 		t,l = IO.unpack_from('<iq',s,offset); offset += 12
// 		ret['index'].append((t,l))
// 	#for f in range(0,len(ret['index']),100): print ret['index'][f]
// 	assert(len(s) == offset)
// 	return ret
}


void undistort_points(object o_x2ds, const float ox, const float oy, const float k1, const float k2, object o_ret) {
	const Np2D<float> x2ds(o_x2ds,-1,-1); // input could be Nx3
	int num2ds = x2ds._rows;
	Np2D<float> ret(o_ret, num2ds, -1);
	for (int i = 0; i < num2ds; ++i) {
		const float *dx2 = x2ds[i];
		float *ux2 = ret[i];
		float dx = dx2[0] - ox, dy = dx2[1] - oy;
		float r2 = dx*dx+dy*dy;
		float sc = (1 + (k1 + k2*r2)*r2);
		ux2[0] = dx * sc + ox;
		ux2[1] = dy * sc + oy;
	}
}

void distort_points(object o_x2ds, const float ox, const float oy, const float k1, const float k2, object o_ret) {
	const Np2D<float> x2ds(o_x2ds,-1,-1); // input could be Nx3
	int num2ds = x2ds._rows;
	Np2D<float> ret(o_ret, num2ds, -1);
	for (int i = 0; i < num2ds; ++i) {
		const float *ux2 = x2ds[i];
		float *dx2 = ret[i];
		float ux = ux2[0] - ox, uy = ux2[1] - oy;
		// ux = dx * sc, uy = dy * sc
		// sc = (1 + (k1 + k2*r2)*r2)
		// r2 = dx*dx+dy*dy
		float dx = ux, dy = uy;
		for (int j = 0; j < 5; ++j) {
			float r2 = dx*dx+dy*dy;
			float sc = (1 + (k1 + k2*r2)*r2);
			dx = ux/sc;
			dy = uy/sc; // update
		}
		dx2[0] = dx + ox;
		dx2[1] = dy + oy;
	}
}


int project_and_clean(object &o_x3ds, object &o_Ps, object &o_x2ds_data, object o_x2ds_splits, object &o_x2ds_labels, object &o_x2ds_labels2,
					  const float thresh_point, const float thresh_camera, const float thresh_x3d) {
	const Np2D<float>   x3ds(o_x3ds, -1, 3);
	const int           num3ds = x3ds._rows;
	const Np2D<float>   x2ds_data(o_x2ds_data, -1, 2);
	const int           num2ds = x2ds_data._rows;
	const Np1D<int32_t> x2ds_splits(o_x2ds_splits, -1);
	const int           numCams = x2ds_splits._size-1;
	Np1D<int32_t>       x2ds_labels(o_x2ds_labels, num2ds); // may be modified
	Np1D<int32_t>       x2ds_labels2(o_x2ds_labels2, num2ds); // may be modified
	const Np3D<float> Ps(o_Ps, numCams, 3, 4);
	int goodCams = 0;
	for (int ci = 0; ci < numCams; ++ci) {
		int c0 = x2ds_splits[ci], c1 = x2ds_splits[ci+1];
		const float *P = Ps[ci];
		float rms = -1;
		for (int di = c0; di < c1; ++di) {
			const int li = x2ds_labels2[di];
			const float *x2 = x2ds_data[di];
			if (li != -1) { // projecting point li should align with x2
				assert(li < num3ds);
				const float *xl = x3ds[li];
				const float tmp0 = P[0]*xl[0] + P[1]*xl[1] + P[2]*xl[2] + P[3];
				const float tmp1 = P[4]*xl[0] + P[5]*xl[1] + P[6]*xl[2] + P[7];
				const float tmp2 = P[8]*xl[0] + P[9]*xl[1] + P[10]*xl[2] + P[11];
				const float sc = (tmp2 != 0 ? -1.0/tmp2 : 0.0);
				const float dx = tmp0*sc - x2[0];
				const float dy = tmp1*sc - x2[1];
				const float r = dx*dx+dy*dy;
				if (r >= thresh_point) { x2ds_labels[di] = x2ds_labels2[di] = -1; }
				if (thresh_x3d > 0) {
					if (-tmp2 < 0 || -tmp2 > thresh_x3d) { x2ds_labels[di] = x2ds_labels2[di] = -1; }
				}
				rms = std::max(rms, r);
			}
		}
		if (rms >= thresh_camera) {
			for (int di = c0; di < c1; ++di) {
				x2ds_labels2[di] = -1;
				x2ds_labels[di] = -1;
			}
		}
		else if (rms != -1) {
			goodCams++;
		}
	}
	return goodCams;
}

float project_and_compute_rms(object &o_x3ds, object o_x2ds, object &o_P, const float ox, const float oy, const float k1, const float k2) {
	const Np2D<float> x2ds(o_x2ds,-1,2);
	int num2ds = x2ds._rows;
	const Np2D<float> x3ds(o_x3ds, num2ds, 3);
	const Np2D<float> Pmat(o_P, 3, 4);
	const float *P = Pmat._data;
	float sse = 0;
	for (int i = 0; i < num2ds; ++i) {
		const float *xl = x3ds[i];
		const float tmp0 = P[0]*xl[0] + P[1]*xl[1] + P[2]*xl[2] + P[3];
		const float tmp1 = P[4]*xl[0] + P[5]*xl[1] + P[6]*xl[2] + P[7];
		const float tmp2 = P[8]*xl[0] + P[9]*xl[1] + P[10]*xl[2] + P[11];
		const float *x2 = x2ds[i];
		const float x = x2[0] - ox, y = x2[1] - oy;
		const float r2 = x*x+y*y;
		const float sc = (1 + (k1 + k2*r2)*r2);
		const float od0 = x * sc + ox;
		const float od1 = y * sc + oy;
		const float sc2 = (tmp2 != 0 ? -1.0/tmp2 : 0.0);
		const float dx = tmp0*sc2 - od0;
		const float dy = tmp1*sc2 - od1;
		sse += dx*dx+dy*dy;
	}
	return sqrtf(sse/num2ds);
}

object sum_indices(object &o_mat, object &o_indices) {
	const Np2D<float> mat(o_mat);
	const int matRows = mat._rows, vecSize = mat._cols;
	const Np1D<int32_t> indices(o_indices);
	const int indSize = indices._size;
	std::vector<int> indScl(indSize);
	for (int i = 0; i < indSize; ++i) indScl[i] = indices[i]*vecSize;
	std::vector<float> out(vecSize);
	//#pragma omp parallel for
	for (int j = 0; j < vecSize; ++j) {
		double sum = 0;
		const float *md = mat._data + j;
		for (int i = 0; i < indSize; ++i) sum += md[indScl[i]];
		out[j] = float(sum);
	}
	return newArrayFromVector<float>(out);
}

void traverse_forest(object &o_splits, object &o_pixels, object &o_leaf_indices) {
	const Np3D<int32_t> splits(o_splits,-1,-1,3);
	const int numTrees = splits._rows;
	const int treeDepth = splits._cols;
	const Np1D<unsigned char> pixels(o_pixels, -1);
	const unsigned char *fpv = pixels._data;
	Np1D<int32_t> leaf_indices(o_leaf_indices,numTrees);
	int32_t *lp = leaf_indices._data;
	for (int ti = 0; ti < numTrees; ++ti) {
		const int32_t *sp = splits[ti];
		int32_t li = 0;
		while (li < treeDepth) {
			const int32_t *spli = sp+3*li;
			const int32_t idx1 = spli[0], idx2 = spli[1], thresh = spli[2];
			li = 2*li+((fpv[idx1] < thresh + fpv[idx2])?2:1);
		}
		lp[ti] = (li-treeDepth) + ti*(treeDepth+1);
	}
}

void get_affine_mat(const int numVerts, const float *pinv, const float *shape, float *amat) {
	float m00=0,m01=0,m10=0,m11=0;
	const float *shape_i = shape;
	const float *pinv_x = pinv, *pinv_y = pinv+numVerts;
	for (int i = 0; i < numVerts; ++i) {
		const float px = *pinv_x++;
		const float py = *pinv_y++;
		const float sx = *shape_i++;
		const float sy = *shape_i++;
		m00 += px*sx; m01 += px*sy;
		m10 += py*sx; m11 += py*sy;
	}
	amat[0] = m00;
	amat[1] = m01;
	amat[2] = m10;
	amat[3] = m11;
}

void pixel_sample(const float *amat, const int numVerts, const float *shape,
				  const int numPixels, const int32_t *rpas, const float *rpds,
				  const int height, const int width, const int chans, const unsigned char *img,
				  unsigned char *pix, bool flipy)
{
	const float m00 = amat[0], m01 = amat[1], m10 = amat[2], m11 = amat[3];
	for (int i = 0; i < numPixels; ++i) {
		const float *c = shape+2*(*rpas++);
		const float dx = *rpds++;
		const float dy = *rpds++;
		float x = (c[0] + dx*m00 + dy*m10);
		float y = (c[1] + dx*m01 + dy*m11);
		if (flipy) y = height-y;
		if (x < 0) x = 0; if (x >= width ) x = width-1;
		if (y < 0) y = 0; if (y >= height) y = height-1;
		pix[i] = img[(int(y)*width + int(x))*chans];
	}
}

void sample_pixels(object &o_shape, object &o_ref_pinv, object &o_anchors, object &o_deltas, object &o_img, object &o_pixels, bool flipy) {
	Np1D<unsigned char> pixels(o_pixels, -1);
	unsigned char *fpv = pixels._data;

	const Np2D<float> shape(o_shape,-1,2);
	const int numVerts = shape._rows;
	const Np2D<float> ref_pinv(o_ref_pinv,2,numVerts);
	const Np1D<int32_t> anchors(o_anchors,-1);
	const int numPixels = anchors._size;
	assert(numPixels > 0);
	const Np2D<float> deltas(o_deltas,numPixels,2);
	const Np3D<unsigned char> img(o_img, -1, -1, -1);
	const int height = img._rows, width = img._cols, chans = img._chans;
	float amat[4];
	get_affine_mat(numVerts, ref_pinv._data, shape._data, amat);
	pixel_sample(amat, numVerts, shape._data, numPixels, anchors._data, deltas._data, height, width, chans, img._data+chans/2, fpv, flipy);
}

void eval_shape_predictor(object &o_ref_pinv, object &o_ref_shape, object &o_splits, object &o_leaves, object &o_anchors, object &o_deltas, object &o_img, object &o_shape, bool flipy) {
	Np2D<float> shape(o_shape,-1,2);
	const int numVerts = shape._rows;
	const Np2D<float> ref_pinv(o_ref_pinv,2,numVerts);
	const Np2D<float> ref_shape(o_ref_shape,numVerts,2);
	const Np4D<int32_t> splits(o_splits,-1,-1,-1,3);
	const int numEpochs = splits._items;
	const int numTrees = splits._rows;
	const int treeDepth = splits._cols;
	const Np4D<float> leaves(o_leaves,numEpochs,numTrees,treeDepth+1,numVerts*2);
	const int limax = leaves._cols; // =treeDepth+1
	const Np2D<int32_t> anchors(o_anchors,numEpochs,-1);
	const int numPixels = anchors._cols;
	assert(numPixels > 0);
	const Np3D<float> deltas(o_deltas,numEpochs,numPixels,2);
	const Np3D<unsigned char> img(o_img, -1, -1, -1);
	const int height = img._rows, width = img._cols, chans = img._chans;
	memcpy(shape._data, ref_shape._data, sizeof(float)*numVerts*2);
	std::vector<unsigned char> tmp(numPixels);
	unsigned char * const pix = &tmp[0];
	std::vector<int> indScl(numTrees);
	for (int ei = 0; ei < numEpochs; ++ei) {
		const float *le = leaves[ei];
		const int32_t *spe = splits[ei];
		float amat[4];
		get_affine_mat(numVerts, ref_pinv._data, shape._data, amat);
		const float m00 = amat[0], m01 = amat[1], m10 = amat[2], m11 = amat[3];
		pixel_sample(amat, numVerts, shape._data, numPixels, anchors[ei], deltas[ei], 
					 height, width, chans, img._data+chans/2, pix, flipy);
		for (int ti = 0; ti < numTrees; ++ti) {
			const int32_t *sp = spe+ti*(treeDepth*3);
			int32_t li = 0;
			while (li < treeDepth) {
				const int32_t *spli = sp+3*li;
				const int32_t idx0 = spli[0], idx1 = spli[1], thresh = spli[2];
				li = 2*li+((pix[idx0] < thresh + pix[idx1])?2:1);
			}
			indScl[ti] = (ti*limax + (li-treeDepth))*(numVerts*2);
		}
		//#pragma omp parallel for
		for (int i = 0; i < numVerts; ++i) {
			double x = 0, y = 0;
			const float *ld = le + 2*i;
			for (int ti = 0; ti < numTrees; ++ti) { const float *ldd = ld+indScl[ti]; x += *ldd++; y += *ldd; }
			shape[i][0] += m00 * x + m10 * y;
			shape[i][1] += m01 * x + m11 * y;
		}
	}
}

object trianglesToEdgeList(object o_triangles, int numVerts = 0) {
	const Np2D<int32_t> triangles(o_triangles,-1,3);
	// Convert a list of triangle indices to an array of up-to-10 neighbouring vertices per vertex (following anticlockwise order).
	if (numVerts < 1) numVerts = triangles.max(0)+1;
	typedef std::map<int32_t,int32_t> imap;
	typedef std::vector<imap> vmap;
	vmap T = vmap(numVerts);
	vmap P = vmap(numVerts);
	const int numTriangles = triangles._rows;
	for (int ti = 0; ti < numTriangles; ++ti) {
		const int32_t *td = triangles[ti];
		int32_t t0 = td[0],t1 = td[1],t2 = td[2];
		T[t0][t1] = t2; T[t1][t2] = t0; T[t2][t0] = t1;
		P[t1][t0] = t2; P[t2][t1] = t0; P[t0][t2] = t1;
	}
	object o_S(newArray2D<int32_t>(numVerts, 10));
	Np2D<int32_t> S(o_S, numVerts, 10);
	for (int vi = 0; vi < numVerts; ++vi) {
		int32_t *Si = S[vi];
		imap &es = T[vi];
		imap &ps = P[vi];
		for (int i = 0; i < 10; ++i) Si[i] = vi;
		if (es.size() == 0) continue;
		int32_t v = es.begin()->first;
		imap::iterator p,e;
		while ((p = ps.find(v)) != ps.end()) { v = p->second; ps.erase(p); }
		for (int li = 0; li < 10; ++li) {
			Si[li] = v;
			e = es.find(v);
			if (e == es.end()) break;
			v = e->second;
			es.erase(e);
		}
	}
	return o_S;
}

BOOST_PYTHON_MODULE(ISCV)
{
	import_array();
	numeric::array::set_module_and_type("numpy", "ndarray");

	class_<Dot>("Dot")
		.def_readwrite("x0", &Dot::x0)
		.def_readwrite("x1", &Dot::x1)
		.def_readwrite("y0", &Dot::y0)
		.def_readwrite("y1", &Dot::y1)
		.def_readwrite("sx", &Dot::sx)
		.def_readwrite("sy", &Dot::sy)
		.def_readwrite("sxx", &Dot::sxx)
		.def_readwrite("sxy", &Dot::sxy)
		.def_readwrite("syy", &Dot::syy)
	;

	class_<std::vector<Dot> >("DotVec")
	.def(vector_indexing_suite<std::vector<Dot> >());


	def("filter_image", &filter_image); 
 
	def("detect_bright_dots",&detect_bright_dots);
	def("detect_bright_dots",&detect_bright_dots_box);
	def("pose_skeleton",&pose_skeleton);
	def("pose_skeleton_with_chan_mats",&pose_skeleton_with_chan_mats);
	def("copy_joints",&copy_joints);
	def("bake_ball_joints",&bake_ball_joints);
	def("unbake_ball_joints",&unbake_ball_joints);
	def("pose_effectors",&pose_effectors);
	def("pose_effectors_single_ray",&pose_effectors_single_ray);
	def("score_effectors",&score_effectors);
	def("marker_positions",&marker_positions);
	def("derror_dchannel",&derror_dchannel);
	def("derror_dchannel_single_ray",&derror_dchannel_single_ray);
	def("line_search",&line_search);
	def("compute_E",&compute_E);
	def("dot",&dot);
	def("dets_to_rays",&dets_to_rays);
	def("intersect_rays",&intersect_rays);
	def("intersect_rays2",&intersect_rays2);
	def("intersect_rays_base", &intersect_rays_base);
	def("solve_x3ds", &solve_x3ds);
	def("solve_x3ds_rays", &solve_x3ds_rays);
	def("project",&project);
	def("project_visibility", &project_visibility);
	def("update_vels",&update_vels);
	def("J_transpose",&J_transpose);
	def("JTJ",&JTJ);
	def("JTJ_single_ray",&JTJ_single_ray);
	def("min_assignment_sparse",&min_assignment_sparse);
	def("label_from_graph",&label_from_graph);
	def("dm_from_l3ds",&dm_from_l3ds);
	def("label_T_wand",&label_T_wand);
	def("decode_X2D",&decode_X2D);
	def("undistort_points",&undistort_points);
	def("distort_points",&distort_points);
	def("project_and_clean",&project_and_clean);
	def("project_and_compute_rms",&project_and_compute_rms);
	def("sum_indices",&sum_indices);
	def("traverse_forest",&traverse_forest);
	def("sample_pixels",&sample_pixels);
	def("eval_shape_predictor",&eval_shape_predictor);
	def("trianglesToEdgeList",&trianglesToEdgeList);

	class_<HashCloud2D>("HashCloud2D", init<object&,float>())
		.def("score", &HashCloud2D::score)
	;

	class_<HashCloud2DList>("HashCloud2DList", init<object&,object&,float>())
		.def("assign", &HashCloud2DList::assign)
		.def("assign_with_vel", &HashCloud2DList::assign_with_vel)
		.def("propagate", &HashCloud2DList::propagate)
		.def("project_assign", &HashCloud2DList::project_assign)
		.def("project_assign_visibility", &HashCloud2DList::project_assign_visibility)
	;

	class_<HashCloud3D>("HashCloud3D", init<object&,float>())
		.def("score", &HashCloud3D::score)
	;

	class_<ProjectVisibility, boost::shared_ptr<ProjectVisibility> >("ProjectVisibility")
		.def("create", &ProjectVisibility::create).staticmethod("create")
		.def("setNormals", &ProjectVisibility::setNormals)
		.def("setLods", &ProjectVisibility::setLods)
		.def("setNormalsAndLods", &ProjectVisibility::setNormalsAndLods)
	;
}

} // namespace ISCV_python
#pragma warning(pop)
