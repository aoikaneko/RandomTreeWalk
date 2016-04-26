#include <Eigen/Core>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <myutil/kinect_handler.h>
#include "kinect_handler.h"

/**
* @brief Set image resolution for coordinate mapping.
* @param resolution Image resolution. '320x240' and '640x480' are supported.
*/
void PyKinect::set_resolution(std::string resolution)
{
	if (resolution == "320x240"){
		rows_ = 240;
		cols_ = 320;
		image_resolution_ = NUI_IMAGE_RESOLUTION_320x240;
	}
	else if (resolution == "640x480"){
		rows_ = 480;
		cols_ = 640;
		image_resolution_ = NUI_IMAGE_RESOLUTION_640x480;
	}
	else{
		throw std::exception("Argument error in Pykinect::set_resolution. Use '320x240' or '640x480'.");
	}
}

/**
* @brief Get current image resolution for coordinate mapping.
* @return resolution Current image resolution. '320x240' or '640x480'.
*/
std::string PyKinect::get_resolution(void) const
{
	std::string resolution;

	if (image_resolution_ == NUI_IMAGE_RESOLUTION_320x240){
		resolution = "320x240";
	}
	else{
		resolution = "640x480";
	}

	return resolution;
}

/**
* @brief Project 3D points in world coordinate into image plane.
* @param points3d 3D points in world coordinates. The type should be (n x 3) numpy::ndarray.
* @return image_points Output 2D points on the image plane. The type is (n x 2) numpy::ndarray.
*/
np::ndarray PyKinect::project_points3d(np::ndarray& points3d) const
{
	int n_data = static_cast<int>(points3d.shape(0));
	int dims_3d = 3;

	if (points3d.get_nd() != 2 || points3d.shape(1) != dims_3d || points3d.get_dtype() != np::dtype::get_builtin<double>()){
		throw std::runtime_error("points3d should be (n x 3) numpy::ndarray. The type should be float64.");
	}

	int nd = 2;
	Py_intptr_t shape[2] = { n_data, 2 };
	np::ndarray image_points = np::zeros(nd, shape, np::dtype::get_builtin<double>());

	auto strides_3d = points3d.get_strides();
	auto strides_2d = image_points.get_strides();
	for (int i = 0; i < n_data; ++i)
	{
		double* tmp = reinterpret_cast<double*>(points3d.get_data() + i * strides_3d[0]);

		float x, y;
		Eigen::Vector3f point(tmp[0], tmp[1], tmp[2]);
		myutil::KinectSDKHandler::project3DPointToImagePlane(point, x, y, image_resolution_);

		*reinterpret_cast<double*>(image_points.get_data() + i * strides_2d[0] + 0 * strides_2d[1]) = static_cast<double>(x);
		*reinterpret_cast<double*>(image_points.get_data() + i * strides_2d[0] + 1 * strides_2d[1]) = static_cast<double>(y);
	}

	return image_points;
}

/**
* @brief Create depth map from 3D point cloud by projecting 3D points into image plane.
* @param pointcloud 3D point cloud in world coordinates. The type should be (n x 3) numpy::ndarray.
* @return depthmap Output 2D depth map that containes distance values in mm. The type is (480 x 640) numpy::ndarray.
*/
np::ndarray PyKinect::create_depthmap(np::ndarray& pointcloud) const
{
	int n_data = static_cast<int>(pointcloud.shape(0));
	int dims_3d = 3;

	if (pointcloud.get_nd() != 2 || pointcloud.shape(1) != dims_3d || pointcloud.get_dtype() != np::dtype::get_builtin<double>()){
		throw std::runtime_error("pointcloud should be (n x 3) numpy::ndarray. The type should be float64.");
	}

	int rows = rows_;
	int cols = cols_;

	int nd = 2;
	Py_intptr_t shape[2] = { rows, cols };
	np::ndarray depthmap = np::zeros(nd, shape, np::dtype::get_builtin<ushort>());

	auto strides_depth = depthmap.get_strides();

	// fill depthmap with ushort::max()
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int y = 0; y < rows; ++y)
	{
		ushort* tmp = reinterpret_cast<ushort*>(depthmap.get_data() + y * strides_depth[0]);

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int x = 0; x < cols; ++x){
			tmp[x] = std::numeric_limits<ushort>::max();
		}
	}

	auto strides_cloud = pointcloud.get_strides();

	for (int i = 0; i < n_data; ++i)
	{
		double* tmp = reinterpret_cast<double*>(pointcloud.get_data() + i * strides_cloud[0]);

		if (_isnan(tmp[0]) || _isnan(tmp[1]) || _isnan(tmp[2])){
			continue;
		}

		float x_f, y_f;
		Eigen::Vector3f point(tmp[0], tmp[1], tmp[2]);
		myutil::KinectSDKHandler::project3DPointToImagePlane(point, x_f, y_f, image_resolution_);

		int x = static_cast<int>(std::floor(x_f + 0.5));
		int y = static_cast<int>(std::floor(y_f + 0.5));

		if ((x < 0 || rows - 1 < x) || (y < 0 || cols - 1 < y)){
			continue;
		}

		// meters -> mm
		ushort depth = static_cast<ushort>(std::floor(tmp[2] * 1000.0 + 0.5));

		*reinterpret_cast<ushort*>(depthmap.get_data() + y * strides_depth[0] + x * strides_depth[1]) = depth;
	}

	// set empty pixels to zero
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int y = 0; y < rows; ++y)
	{
		ushort* tmp = reinterpret_cast<ushort*>(depthmap.get_data() + y * strides_depth[0]);

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int x = 0; x < cols; ++x)
		{
			if (tmp[x] == std::numeric_limits<ushort>::max()){
				tmp[x] = 0;
			}
		}
	}

	return depthmap;
}

/**
* @brief Reproject depth pixels to the 3D points.
* @param depth_pixels Depth pixels containing x, y image coordinates (in pixels) and depth values (in mm). The type should be (n x 3) numpy::ndarray.
* @return points_3d Reprojected points in world coordinates (in meters). The type is (n x 3) numpy::ndarray.
*/
np::ndarray PyKinect::reproject_image_points(np::ndarray& depth_pixels) const
{
	int n_data = static_cast<int>(depth_pixels.shape(0));
	int dims = 3;

	if (depth_pixels.get_nd() != 2 || depth_pixels.shape(1) != dims || depth_pixels.get_dtype() != np::dtype::get_builtin<double>()){
		throw std::runtime_error("depth_pixels should be (n x 3) numpy::ndarray. The type should be float64.");
	}

	int nd = 2;
	Py_intptr_t shape[2] = { n_data, 3 };
	np::ndarray points3d = np::zeros(nd, shape, np::dtype::get_builtin<double>());

	auto strides_depth = depth_pixels.get_strides();
	auto strides_3d = points3d.get_strides();
	for (int i = 0; i < n_data; ++i)
	{
		double* tmp = reinterpret_cast<double*>(depth_pixels.get_data() + i * strides_depth[0]);

		int x = static_cast<int>(std::floor(tmp[0] + 0.5));
		int y = static_cast<int>(std::floor(tmp[1] + 0.5));
		ushort depth = static_cast<ushort>(std::floor(tmp[0] + 0.5));

		Eigen::Vector3f point;
		myutil::KinectSDKHandler::convertDepthPixelTo3DPoint(x, y, depth, point, image_resolution_);

		*reinterpret_cast<double*>(points3d.get_data() + i * strides_3d[0] + 0 * strides_3d[1]) = static_cast<double>(point.x());
		*reinterpret_cast<double*>(points3d.get_data() + i * strides_3d[0] + 1 * strides_3d[1]) = static_cast<double>(point.y());
		*reinterpret_cast<double*>(points3d.get_data() + i * strides_3d[0] + 2 * strides_3d[1]) = static_cast<double>(point.z());
	}

	return points3d;
}

/**
* @brief Convert depth map to 3D point cloud.
* @param depth_map Input depth map. The type should be (480 x 640) numpy::ndarray with dtype=uint16.
* @return cloud Converted 3D points in world coordinates (in meters). The type is (n x 3) numpy::ndarray with dtype=float64.
*/
np::ndarray PyKinect::create_pointcloud(np::ndarray& depth_map) const
{
	int rows = rows_;
	int cols = cols_;

	int n_data = static_cast<int>(depth_map.shape(0) * depth_map.shape(1));

	if (depth_map.get_nd() != 2 || depth_map.shape(0) != rows || depth_map.shape(1) != cols || depth_map.get_dtype() != np::dtype::get_builtin<ushort>()){
		std::stringstream error_message;
		error_message << "depth_map should be " << rows << "x" << cols << " numpy::ndarray with dtype=uint16.";
		throw std::runtime_error(error_message.str());
	}

	auto strides_depth = depth_map.get_strides();

	pcl::PointXYZ default_value(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>(cols, rows, default_value));

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int y = 0; y < rows; ++y)
	{
		ushort* tmp = reinterpret_cast<ushort*>(depth_map.get_data() + y * strides_depth[0]);

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int x = 0; x < cols; ++x)
		{
			int idx = y * cols + x;

			ushort z = tmp[x];

			if (z != 0)
			{
				Eigen::Vector3f point;
				myutil::KinectSDKHandler::convertDepthPixelTo3DPoint(x, y, z, point, image_resolution_);

				pcl_cloud->points.at(idx) = pcl::PointXYZ(point.x(), point.y(), point.z());
			}
		}
	}

	// Pass through filter for z-direction
	pcl::PassThrough<pcl::PointXYZ> pass_through;
	pass_through.setFilterFieldName("z");
	pass_through.setFilterLimits(1.0, 10.0);
	pass_through.setKeepOrganized(true);
	pass_through.setInputCloud(pcl_cloud);
	pass_through.filter(*pcl_cloud);

	// Statistical filter
	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> outlier_removal;
	outlier_removal.setInputCloud(pcl_cloud);
	outlier_removal.setMeanK(50);
	outlier_removal.setStddevMulThresh(1.0);
	outlier_removal.setKeepOrganized(true);
	outlier_removal.filter(*pcl_cloud);

	int nd = 2;
	Py_intptr_t shape[2] = { n_data, 3 };
	np::ndarray cloud = np::zeros(nd, shape, np::dtype::get_builtin<double>());

	auto strides_cloud = cloud.get_strides();

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < n_data; ++i)
	{
		if (!_isnan(pcl_cloud->points[i].x))
		{
			*reinterpret_cast<double*>(cloud.get_data() + i * strides_cloud[0] + 0 * strides_cloud[1]) = static_cast<double>(pcl_cloud->points[i].x);
			*reinterpret_cast<double*>(cloud.get_data() + i * strides_cloud[0] + 1 * strides_cloud[1]) = static_cast<double>(pcl_cloud->points[i].y);
			*reinterpret_cast<double*>(cloud.get_data() + i * strides_cloud[0] + 2 * strides_cloud[1]) = static_cast<double>(pcl_cloud->points[i].z);
		}
		else
		{
			*reinterpret_cast<double*>(cloud.get_data() + i * strides_cloud[0] + 0 * strides_cloud[1]) = std::numeric_limits<double>::quiet_NaN();
			*reinterpret_cast<double*>(cloud.get_data() + i * strides_cloud[0] + 1 * strides_cloud[1]) = std::numeric_limits<double>::quiet_NaN();
			*reinterpret_cast<double*>(cloud.get_data() + i * strides_cloud[0] + 2 * strides_cloud[1]) = std::numeric_limits<double>::quiet_NaN();
		}
	}

	return cloud;
}