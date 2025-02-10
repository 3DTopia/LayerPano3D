#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <python_binding.hpp>

#include <opencv2/opencv.hpp>

#include <vector>
#include <iostream>
#include <math.h>

//#undef _DEBUG
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

static PyObject* cv2numpy(const cv::Mat& mat_array_)
{
	cv::Mat mat_array;
	// check cv mat is double
	if (mat_array_.depth() != CV_64F)
	{
		//std::cout << "convert CV mat to float64" << std::endl;
		mat_array_.convertTo(mat_array, CV_64F);
	}
	else {
		mat_array = mat_array_;
	}

	//
	int nbRow = mat_array.rows;
	int nbCol = mat_array.cols;

	//Set the size of the numpy array
	npy_intp dims[2];
	dims[0] = nbRow;
	dims[1] = nbCol;
	double* mat = (double*)mat_array.data;
	if (mat == NULL)
	{
		PyObject* objMat = PyArray_EMPTY(2, dims, NPY_FLOAT64, 0);
		if (objMat == NULL)
		{
			PyErr_SetString(PyExc_RuntimeError, "allocMatrix : Could not allocated memory\n");
			return NULL;
		}
		return objMat;
	}

	PyArray_Descr* descr = PyArray_DescrFromType(NPY_FLOAT64);

	int dim_number = 2;

	PyArray_Dims strides = { NULL, 0 };
	strides.len = 2;
	strides.ptr = PyDimMem_NEW(2);
	strides.ptr[1] = sizeof(double); // element size
	strides.ptr[0] = (nbCol)*strides.ptr[1];

	PyObject* objMat = PyArray_NewFromDescr(&PyArray_Type, descr, dim_number, dims, strides.ptr, NULL, NPY_ARRAY_WRITEABLE, NULL);

	if (!objMat)
	{
		if (PyErr_Occurred())
		{
			PyErr_Print();
			PyErr_Clear();
			Py_XDECREF(objMat);
		}
	}
	void* numpy_data = PyArray_DATA((PyArrayObject*)objMat);
	memcpy(numpy_data, (void*)mat_array.data, sizeof(double) * nbRow * nbCol);
	//Py_INCREF(objMat);
	return objMat;
}

static int numpy2cv(PyObject* numpy_array_, cv::Mat& mat_array)
{
	if (numpy_array_ == NULL)
	{
		PyErr_SetString(PyExc_RuntimeError, "Object is NULL!\n");
		return -1;
	}
	if (PyArray_Check(numpy_array_) == false)
	{
		std::cout << "Object is not numpy object" << std::endl;
		PyErr_SetString(PyExc_RuntimeError, "Object is not numpy object!\n");
		return -1;
	}

	PyArrayObject* numpy_array = NULL;

	// check numpy data type
	int obj_type = PyArray_TYPE((PyArrayObject*)numpy_array_);
	if (NPY_FLOAT64 != obj_type)
	{
		PyErr_SetString(PyExc_RuntimeError, "The depth map data type should be float64.\n");
		return -1;
		//return cv::Mat();
		//std::cout << "Convert the data to float64 type" << std::endl;
		//numpy_array = (PyArrayObject*)PyArray_CastToType((PyArrayObject*)numpy_array_, PyArray_DescrFromType(NPY_FLOAT64), 0);
		//numpy_array = PyArray_FROM_OTF(numpy_array_, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS);
	}
	//else {
	//    numpy_array  = (PyArrayObject*)(numpy_array_);
	//    //numpy_array = PyArray_FROM_OTF(numpy_array_, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS);
	//}

	numpy_array = (PyArrayObject*)PyArray_FROM_OTF(numpy_array_, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	if (numpy_array == NULL)
		return -1;

	int ndim = PyArray_NDIM(numpy_array);

	if (ndim != 2)
	{
		PyErr_SetString(PyExc_RuntimeError, "The depth map size should be [height, width].\n");
		return -1;
	}
	//
	npy_intp* array_shape = PyArray_SHAPE(numpy_array);
	int row_number = (int)array_shape[0];
	int col_number = (int)array_shape[1];

	char* data = (char*)PyArray_DATA(numpy_array);
	mat_array = cv::Mat(row_number, col_number, CV_64FC1);
	memcpy(mat_array.data, data, sizeof(double) * row_number * col_number * 1);

	Py_DECREF(numpy_array);

	return 0;
}

static PyObject* create_debug_data(PyObject* self, PyObject* args)
{
	std::vector<cv::Mat> depthmap;
	std::vector<cv::Mat> align_coeff;
	std::map<int, std::map<int, cv::Mat>> pixels_corresponding_list;

	int debug_data_type;
	int frame_number;

	if (!PyArg_ParseTuple(args, "ii",
		&debug_data_type,
		&frame_number))
	{
		PyErr_SetString(PyExc_RuntimeError, "Input data parse error!\n");
		return NULL;
	}

	create_debug_data(depthmap,
		align_coeff,
		pixels_corresponding_list,
		debug_data_type,
		frame_number
	);

	// check the return value
	if (depthmap.size() == 0)
	{
		PyErr_SetString(PyExc_RuntimeError, "The generated depth map list is empty!\n");
		return NULL;
	}
	if (align_coeff.size() == 0)
	{
		PyErr_SetString(PyExc_RuntimeError, "The generated depth map coefficients list is empty!\n");
		return NULL;
	}

	// encapsulate to python objects
	PyObject* depthmap_list_py = PyList_New(depthmap.size());
	for (long unsigned int index = 0; index < depthmap.size(); index++)
	{
		if (depthmap[index].size().area() == 0)
		{
			PyErr_SetString(PyExc_RuntimeError, "The depth map generation error!\n");
			return NULL;
		}
		PyList_SetItem(depthmap_list_py, index, cv2numpy(depthmap[index]));
	}

	PyObject* align_coeff_py = PyList_New(align_coeff.size());
	for (long unsigned int index = 0; index < align_coeff.size(); index++)
	{
		if (align_coeff[index].size().area() == 0)
		{
			PyErr_SetString(PyExc_RuntimeError, "The align coefficients generation error!\n");
			return NULL;
		}
		PyList_SetItem(align_coeff_py, index, cv2numpy(align_coeff[index]));
	}

	PyObject* pixels_corresponding_list_py = PyDict_New();
	for (auto iter_src = pixels_corresponding_list.begin(); iter_src != pixels_corresponding_list.end(); ++iter_src)
	{
		int key_src = iter_src->first;
		std::map<int, cv::Mat>& value_tar = pixels_corresponding_list[key_src];

		PyObject* value_tar_py = PyDict_New();
		for (auto iter_tar = value_tar.begin(); iter_tar != value_tar.end(); ++iter_tar)
		{
			int key_tar = iter_tar->first;
			cv::Mat& pixel_corr = value_tar[key_tar];

			PyObject* key_tar_py = PyLong_FromLong(key_tar);
			PyDict_SetItem(value_tar_py, key_tar_py, cv2numpy(pixel_corr));
		}
		PyObject* key_src_py = PyLong_FromLong(key_src);
		PyDict_SetItem(pixels_corresponding_list_py, key_src_py, value_tar_py);
	}

	// return
	return PyTuple_Pack(3, depthmap_list_py, align_coeff_py, pixels_corresponding_list_py);
}

static PyObject* ceres_solver_option(PyObject* self, PyObject* args)
{
	int num_threads;
	int max_num_iterations;
	int max_linear_solver_iterations;
	int min_linear_solver_iterations;

	if (!PyArg_ParseTuple(args, "iiii",
		&num_threads,
		&max_num_iterations,
		&max_linear_solver_iterations,
		&min_linear_solver_iterations))
	{
		PyErr_SetString(PyExc_RuntimeError, "Input data parse error!\n");
		return NULL;
	}
	int result = solver_params(num_threads, max_num_iterations, max_linear_solver_iterations, min_linear_solver_iterations);
	return PyLong_FromLong(result);
}


static PyObject* depthmap_stitch(PyObject* self, PyObject* args)
{
	const char* root_dir;               // str

	PyObject* terms_weight;             // list[float]
	PyObject* depthmap_original_list;    // list[numpy]
	PyObject* depthmap_original_ico_index;// list[int]
	PyObject* pixels_corresponding_map; // dict{int:{int, numpy}}
	PyObject* align_coeff_initial_scale; // list[numpy]
	PyObject* align_coeff_initial_offset; // list[numpy]

	int reference_depthmap_index;        // the reference depth map index of ico sub depth maps.
	int align_coeff_grid_height;        // int
	int align_coeff_grid_width;
	int debug_wait_for_attach;            // 
	int reproj_perpixel_enable;
	int smooth_pergrid_enable;

	if (!PyArg_ParseTuple(args, "sO!O!O!iO!iiiiO!O!i",
		&root_dir,
		&PyList_Type, &terms_weight,
		&PyList_Type, &depthmap_original_list,
		&PyList_Type, &depthmap_original_ico_index,
		&reference_depthmap_index,
		&PyDict_Type, &pixels_corresponding_map,
		&align_coeff_grid_height,
		&align_coeff_grid_width,
		&reproj_perpixel_enable,
		&smooth_pergrid_enable,
		&PyList_Type, &align_coeff_initial_scale,
		&PyList_Type, &align_coeff_initial_offset,
		&debug_wait_for_attach))
	{
		PyErr_SetString(PyExc_RuntimeError, "Input data parse error!\n");
		return NULL;
	}

	// // 
	// std::cout << "debug_wait_for_attach:" << debug_wait_for_attach << std::endl;
	// std::cout << "reference_depthmap_index:" <<reference_depthmap_index << std::endl;
	// std::cout << "align_coeff_grid_height: " << align_coeff_grid_height << std::endl;
	// std::cout << "align_coeff_grid_width: " << align_coeff_grid_width << std::endl;
	// std::cout << "reproj_perpixel_enable: " << reproj_perpixel_enable << std::endl;
	// std::cout << "smooth_pergrid_enable: " << smooth_pergrid_enable << std::endl;

	std::cout << "Parse python data:" << std::endl;
	// 1) parse parameters
	Py_ssize_t terms_weight_size = PyList_Size(terms_weight);
	std::vector<float> terms_weight_cpp;
	//std::cout << "cost function terms weights are: " << std::endl;
	for (int index = 0; index < terms_weight_size; index++)
	{
		PyObject* item = PyList_GetItem(terms_weight, index);
		terms_weight_cpp.push_back((float)PyFloat_AsDouble(item));
		//std::cout << PyFloat_AS_DOUBLE(item) << std::endl;
	}
	if (PyErr_Occurred()) {
		PyErr_SetString(PyExc_RuntimeError, "Terms weight parse error!\n");
		return NULL;
	}

	// sub-image depth maps
	std::cout << "- Parsing sub-image depth maps" << std::endl;
	Py_ssize_t depthmap_original_list_size = PyList_Size(depthmap_original_list);
	Py_ssize_t depthmap_original_index_list_size = PyList_Size(depthmap_original_ico_index);

	if (debug_wait_for_attach > 0)
	{
		std::cout << "Attach me! Input e to exist this process and any other input will continue."; // no flush needed
		char n; std::cin >> n;
		if (n == 'e')
			exit(0);
	}

	if (depthmap_original_list_size != depthmap_original_index_list_size)
	{
		PyErr_SetString(PyExc_RuntimeError, "The depth map size is not equal index list size!\n");
		return NULL;
	}
	std::vector<cv::Mat> depthmap_original_cpp;
	std::vector<int> depthmap_original_index_cpp;
	for (int index = 0; index < depthmap_original_list_size; index++)
	{
		PyObject* mat_index = PyList_GetItem(depthmap_original_ico_index, index);
		depthmap_original_index_cpp.push_back((int)PyLong_AsLong(mat_index));

		PyObject* item = PyList_GetItem(depthmap_original_list, index);
		cv::Mat mat_data;
		//std::cout << mat_data << std::endl;
		if (numpy2cv(item, mat_data) < 0)
		{
			//PyErr_SetString(PyExc_RuntimeError, "The depth map is empty!\n");
			return NULL;
		}
		else
		{
			depthmap_original_cpp.push_back(mat_data);
		}
	}

	// pixels_corresponding_map
	std::cout << "- Parsing pixels_corresponding_map" << std::endl;
	if (PyDict_Check(pixels_corresponding_map) == false)
	{
		PyErr_SetString(PyExc_RuntimeError, "pixels_corresponding_map is not a dictory object!\n");
		return NULL;
	}
	Py_ssize_t dict_length = PyDict_Size(pixels_corresponding_map);
	if (dict_length < (depthmap_original_list_size - 1))
	{
		PyErr_SetString(PyExc_RuntimeError, "pixels_corresponding_map source map length is wrong!\n");
		return NULL;
	}
	std::map<int, std::map<int, cv::Mat>> pixels_corresponding_list;
	PyObject* pixel_corr_srckeys_list = PyDict_Keys(pixels_corresponding_map);
	int pixle_corr_srckeys_size = (int)PyList_Size(pixel_corr_srckeys_list);
	for (int src_index = 0; src_index < pixle_corr_srckeys_size; src_index++)
	{
		PyObject* srckey_py = PyList_GetItem(pixel_corr_srckeys_list, Py_ssize_t(src_index));
		long srckey_long = PyLong_AsLong(srckey_py);
		PyObject* pixel_map_tar = PyDict_GetItem(pixels_corresponding_map, srckey_py);
		if (pixel_map_tar == NULL)
		{
			char msg[128];
			sprintf(msg, "The pixel corresponding relationship source index %ld is missing!\n", srckey_long);
			PyErr_SetString(PyExc_RuntimeError, msg);
			continue;
		}

		Py_ssize_t tar_length = PyDict_Size(pixel_map_tar);
		if (tar_length < (depthmap_original_list_size - 1))
		{
			PyErr_SetString(PyExc_RuntimeError, "pixels_corresponding_map tar map length is wrong!\n");
			return NULL;
		}

		// convert target data list
		std::map<int, cv::Mat> pixels_corresponding_list_tar;
		PyObject* pixel_corr_tarkeys_list = PyDict_Keys(pixel_map_tar);
		int pixle_corr_tarkeys_size = (int)PyList_Size(pixel_corr_tarkeys_list);
		for (int tar_index = 0; tar_index < pixle_corr_tarkeys_size; tar_index++)
		{
			PyObject* tarkey_py = PyList_GetItem(pixel_corr_tarkeys_list, Py_ssize_t(tar_index));
			long tarkey_long = PyLong_AsLong(tarkey_py);
			PyObject* map_mat = PyDict_GetItem(pixel_map_tar, tarkey_py);
			if (map_mat == NULL)
			{
				char msg[128];
				sprintf(msg, "The pixel corresponding relationship target index %ld is missing!\n", tarkey_long);
				PyErr_SetString(PyExc_RuntimeError, msg);
				continue;
			}
			cv::Mat mat_data;
			// std::cout << mat_data << std::endl;
			if (numpy2cv(map_mat, mat_data) < 0)
			{
				PyErr_SetString(PyExc_RuntimeError, "The pixel corresponding mat is empty!\n");
				return NULL;
			}
			else
				pixels_corresponding_list_tar[tarkey_long] = mat_data;
		}
		pixels_corresponding_list[srckey_long] = pixels_corresponding_list_tar;
	}

	// alignment coefficients
	std::cout << "- Parsing alignment coefficients" << std::endl;
	Py_ssize_t align_coeff_initial_scales_size = PyList_Size(align_coeff_initial_scale);
	Py_ssize_t align_coeff_initial_offset_size = PyList_Size(align_coeff_initial_offset);
	if (align_coeff_initial_scales_size != align_coeff_initial_offset_size)
	{
		PyErr_SetString(PyExc_RuntimeError, "Alignment coefficient list size is not equal!\n");
		return NULL;
	}

	std::vector<cv::Mat> align_coeff_initial_scale_cpp;
	std::vector<cv::Mat> align_coeff_initial_offset_cpp;
	for (int index = 0; index < align_coeff_initial_scales_size; index++)
	{
		PyObject* scale_item = PyList_GetItem(align_coeff_initial_scale, index);
		cv::Mat scale_mat_data;
		if (numpy2cv(scale_item, scale_mat_data) < 0)
		{
			PyErr_SetString(PyExc_RuntimeError, "Alignment coefficient scale mat parse error!\n");
			return NULL;
		}
		else
			align_coeff_initial_scale_cpp.push_back(scale_mat_data);

		PyObject* offset_item = PyList_GetItem(align_coeff_initial_offset, index);
		cv::Mat offset_mat_data;
		if (numpy2cv(offset_item, offset_mat_data) < 0)
		{
			PyErr_SetString(PyExc_RuntimeError, "Alignment coefficient scale mat parse error!\n");
			return NULL;
		}
		else
			align_coeff_initial_offset_cpp.push_back(offset_mat_data);
	}

	// 2) compute the coefficients
	std::vector<cv::Mat> depthmap_aligned;
	std::vector<cv::Mat> align_coeff;

	std::cout << "Aligning depth maps..." << std::endl;
	depthmap_stitch(
		root_dir,
		terms_weight_cpp,
		depthmap_original_cpp,
		depthmap_original_index_cpp,
		reference_depthmap_index,
		pixels_corresponding_list,
		align_coeff_grid_height,
		align_coeff_grid_width,
		reproj_perpixel_enable,
		smooth_pergrid_enable,
		align_coeff_initial_scale_cpp,
		align_coeff_initial_offset_cpp,
		depthmap_aligned,
		align_coeff);

	std::cout << "return alignment coefficient and depth maps..." << std::endl;
	// 3) return he parameters
	PyObject* depthmap_aligned_py = PyList_New(depthmap_aligned.size());
	for (long unsigned int index = 0; index < depthmap_aligned.size(); index++)
	{
		PyList_SetItem(depthmap_aligned_py, index, cv2numpy(depthmap_aligned[index]));
		//PyList_SetItem(depthmap_aligned_py, index, PyLong_FromLong(index));
		//std::cout << index << depthmap_aligned[index].size() << std::endl;
	}
	//Py_INCREF(depthmap_aligned_py);

	PyObject* align_coeff_py = PyList_New(align_coeff.size());
	for (long unsigned int index = 0; index < align_coeff.size(); index++)
	{
		PyList_SetItem(align_coeff_py, index, cv2numpy(align_coeff[index]));
		//PyList_SetItem(align_coeff_py, index, PyLong_FromLong(index));
		//std::cout << index << align_coeff[index].size() << std::endl;
	}
	//Py_INCREF(depthmap_aligned_py);

	return PyTuple_Pack(2, depthmap_aligned_py, align_coeff_py);
}

static PyObject *init(PyObject *self, PyObject *args)
{
	const char* method;                 // str
	if (!PyArg_ParseTuple(args, "s", &method))
	{
		PyErr_SetString(PyExc_RuntimeError, "Input data parse error!\n");
		return NULL;
	}

	int result = init(method);
	return PyLong_FromLong(result);
}

static PyObject *shutdown(PyObject *self, PyObject *args)
{
	int result = shutdown();
	return PyLong_FromLong(result);
}

static PyObject *report_aligned_depthmap_error(PyObject *self, PyObject *args)
{
	int result = report_aligned_depthmap_error();
	return PyLong_FromLong(result);
}


static PyObject* python_numpy_mat_test(PyObject* self, PyObject* args)
{
	PyArrayObject* numpy_array;

	/*  parse single numpy array argument */
	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &numpy_array))
	{
		PyErr_SetString(PyExc_RuntimeError, "allocMatrix : Could not allocated memory\n");
		return NULL;
	}

	cv::Mat data;
	numpy2cv((PyObject*)numpy_array, data);

	std::cout << "OpenCV Mat : M = " << std::endl
		<< " " << data << std::endl
		<< std::endl;

	PyObject* data_numpy = cv2numpy(data);
	return data_numpy;

	Py_RETURN_NONE;
}

// Module's Function Definition structure
static PyMethodDef depthmap_align_methods[] = {
	{"create_debug_data", create_debug_data, METH_VARARGS, "Generate synthetic data for testing depth map align."},
	{"depthmap_stitch", depthmap_stitch, METH_VARARGS, "align the depth map and return the align depth map and coefficient."},
	{"ceres_solver_option", ceres_solver_option, METH_VARARGS, "Set the Ceres solver option."},
	{"init", init, METH_VARARGS, "Initial depth map alignment module."},
	{"shutdown", shutdown, METH_VARARGS, "Clean the depth map alignment pyton module."},
	{"report_aligned_depthmap_error", report_aligned_depthmap_error, METH_VARARGS, "Report the error between the all subimage's aligned depth maps."},
	{"python_numpy_mat_test", python_numpy_mat_test, METH_VARARGS, "test Numpy data transformation."},
	{NULL, NULL, 0, NULL} };

// Module Definition structure
static struct PyModuleDef depthmapAlignModule = {
	PyModuleDef_HEAD_INIT,
	"depthmapAlign",
	"Align depth maps Module",
	-1,
	depthmap_align_methods };

// Initializes module using
// TODO Numpy universal functions https://numpy.org/doc/stable/reference/ufuncs.html
PyMODINIT_FUNC PyInit_depthmapAlign(void)
{
	PyObject* module;
	module = PyModule_Create(&depthmapAlignModule);
	if (module == NULL)
		return NULL;

	// numpy Initializes
	import_array();
	if (PyErr_Occurred())
		return NULL;
	return module;
}