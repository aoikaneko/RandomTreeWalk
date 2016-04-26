#define BOOST_PYTHON_STATIC_LIB
#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include "kinect_handler.h"

namespace np = boost::numpy;

BOOST_PYTHON_MODULE(pykinect_wrapper){
	Py_Initialize();
	np::initialize();
	boost::python::class_<PyKinect>("PyKinect")
		.def("get_rows", &PyKinect::get_rows)
		.def("get_cols", &PyKinect::get_cols)
		.def("set_resolution", &PyKinect::set_resolution)
		.def("get_resolution", &PyKinect::get_resolution)
		.def("project_points3d", &PyKinect::project_points3d)
		.def("create_depthmap", &PyKinect::create_depthmap)
		.def("reproject_image_points", &PyKinect::reproject_image_points)
		.def("create_pointcloud", &PyKinect::create_pointcloud)
		.def_pickle(PyKinect_pickle_suite());
}