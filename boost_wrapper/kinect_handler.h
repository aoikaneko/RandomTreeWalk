#ifndef _PYTHON_KINECT_V1_H_
#define _PYTHON_KINECT_V1_H_

#define BOOST_PYTHON_STATIC_LIB
#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <string>
#include <myutil/kinect_handler.h>

namespace np = boost::numpy;

class PyKinect
{
public:

	friend struct PyKinect_pickle_suite;

	// constructor
	PyKinect() : rows_(480), cols_(640), image_resolution_(NUI_IMAGE_RESOLUTION_640x480)
	{
	}

	/**
	* @brief Get current image vertical resolution.
	* @return rows Current vertical resolution.
	*/
	int get_rows(void) const{
		return rows_;
	}

	/**
	* @brief Get current image horizontal resolution.
	* @return cols Current horizontal resolution.
	*/
	int get_cols(void) const{
		return cols_;
	}

	/**
	* @brief Set image resolution for coordinate mapping.
	* @param resolution Image resolution. '320x240' and '640x480' are supported.
	*/
	void set_resolution(std::string resolution);

	/**
	* @brief Get current image resolution for coordinate mapping.
	* @return resolution Current image resolution. '320x240' or '640x480'.
	*/
	std::string get_resolution(void) const;

	/**
	* @brief Project 3D points in world coordinate into image plane.
	* @param points3d 3D points in world coordinates. The type should be (n x 3) numpy::ndarray.
	* @return image_points Output 2D points on the image plane. The type is (n x 2) numpy::ndarray.
	*/
	np::ndarray project_points3d(np::ndarray& points3d) const;

	/**
	* @brief Create depth map from 3D point cloud by projecting 3D points into image plane.
	* @param pointcloud 3D point cloud in world coordinates. The type should be (n x 3) numpy::ndarray.
	* @return depthmap Output 2D depth map that containes distance values in mm. The type is (480 x 640) numpy::ndarray.
	*/
	np::ndarray create_depthmap(np::ndarray& pointcloud) const;

	/**
	* @brief Reproject depth pixels to the 3D points.
	* @param depth_pixels Depth pixels containing x, y image coordinates (in pixels) and depth values (in mm). The type should be (n x 3) numpy::ndarray.
	* @return points_3d Reprojected points in world coordinates (in meters). The type is (n x 3) numpy::ndarray.
	*/
	np::ndarray reproject_image_points(np::ndarray& depth_pixels) const;

	/**
	* @brief Convert depth map to 3D point cloud.
	* @param depth_map Input depth map. The type should be (480 x 640) numpy::ndarray with dtype=uint16.
	* @return cloud Converted 3D points in world coordinates (in meters). The type is (n x 3) numpy::ndarray with dtype=float64.
	*/
	np::ndarray create_pointcloud(np::ndarray& depth_map) const;

private:

	int rows_;
	int cols_;

	NUI_IMAGE_RESOLUTION image_resolution_;
};


struct PyKinect_pickle_suite : boost::python::pickle_suite
{
	static boost::python::tuple getinitargs(const PyKinect&){
		return boost::python::make_tuple();
	}

	static boost::python::tuple getstate(boost::python::object object)
	{
		PyKinect const& target = boost::python::extract<PyKinect const&>(object)();

		return boost::python::make_tuple(
			object.attr("__dict__"),
			target.get_rows(),
			target.get_cols(),
			target.get_resolution());
	}

	static void setstate(boost::python::object object, boost::python::tuple state)
	{
		using namespace boost::python;
		PyKinect& target = extract<PyKinect&>(object)();

		if (len(state) != 4)
		{
			PyErr_SetObject(PyExc_ValueError,
				("expected 4-item tuple in call to __setstate__; got %s"
				% make_tuple(state)).ptr()
				);
			throw_error_already_set();
		}

		// restore the object's __dict__
		dict d = extract<dict>(object.attr("__dict__"))();
		d.update(state[0]);

		// restore the internal state of the C++ object
		std::string resolution = extract<std::string>(state[3]);
		if (resolution == "320x240" || resolution == "640x480"){
			target.rows_ = extract<int>(state[1]);
			target.cols_ = extract<int>(state[2]);
			target.set_resolution(resolution);
		}
	}

	static bool getstate_manages_dict(void){
		return true;
	}
};


#endif		// _PYTHON_KINECT_V1_H_