#define BOOST_PYTHON_STATIC_LIB
#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <iostream>

namespace np = boost::numpy;

np::ndarray _calc_partition_c(int n_data, int rows, int cols, np::ndarray& sample_frame_ids, np::ndarray& sample_offset_ids, np::ndarray& depthmaps, np::ndarray& offsets, np::ndarray& t1, np::ndarray& t2, np::ndarray& tau, float depth_background_val)
{
	using ushort = unsigned short;

	if (sample_frame_ids.get_nd() != 1 || sample_frame_ids.get_dtype() != np::dtype::get_builtin<ushort>()){
		throw std::runtime_error("sample_frame_ids should be numpy::ndarray with dtype=np::uint16 and ndim=1.");
	}

	if (sample_offset_ids.get_nd() != 1 || sample_offset_ids.get_dtype() != np::dtype::get_builtin<ushort>()){
		throw std::runtime_error("sample_offset_ids should be numpy::ndarray with dtype=np::uint16 and ndim=1.");
	}

	if (n_data != sample_frame_ids.shape(0) || n_data != sample_offset_ids.shape(0)){
		throw std::runtime_error("n_data should match with the length of sample_frame_ids and sample_offset_ids.");
	}

	if (depthmaps.get_nd() != 3 || depthmaps.get_dtype() != np::dtype::get_builtin<float>()){
		throw std::runtime_error("depthmaps should be numpy::ndarray with dtype=np::float32 and ndim=4.");
	}

	if (offsets.get_nd() != 3 || offsets.get_dtype() != np::dtype::get_builtin<short>()){
		throw std::runtime_error("offsets should be numpy::ndarray with dtype=np::int16 and ndim=3.");
	}

	if (t1.get_nd() != 1 || t1.get_dtype() != np::dtype::get_builtin<float>()){
		throw std::runtime_error("t1 should be numpy::ndarray with dtype=np::float32 and ndim=1.");
	}

	if (t2.get_nd() != 1 || t2.get_dtype() != np::dtype::get_builtin<float>()){
		throw std::runtime_error("t2 should be numpy::ndarray with dtype=np::float32 and ndim=1.");
	}

	if (tau.get_nd() != 1 || tau.get_dtype() != np::dtype::get_builtin<float>()){
		throw std::runtime_error("tau should be numpy::ndarray with dtype=np::float32 and ndim=1.");
	}

	ushort* frame_ids_c = reinterpret_cast<ushort*>(sample_frame_ids.get_data());
	ushort* offset_ids_c = reinterpret_cast<ushort*>(sample_offset_ids.get_data());

	float* t1_c = reinterpret_cast<float*>(t1.get_data());
	float* t2_c = reinterpret_cast<float*>(t2.get_data());
	float tau_c = reinterpret_cast<float*>(tau.get_data())[0];

	int nd = 1;
	Py_intptr_t shape[1] = { n_data };
	np::ndarray partition = np::zeros(nd, shape, np::dtype::get_builtin<bool>());

	auto depth_strides = depthmaps.get_strides();
	auto offset_strides = offsets.get_strides();
	auto partition_strides = partition.get_strides();

	for (int i = 0; i < n_data; ++i)
	{
		ushort frame_id = frame_ids_c[i];
		ushort offset_id = offset_ids_c[i];

		short u = *reinterpret_cast<short*>(offsets.get_data() + frame_id * offset_strides[0] + offset_id * offset_strides[1] + 0 * offset_strides[2]);
		short v = *reinterpret_cast<short*>(offsets.get_data() + frame_id * offset_strides[0] + offset_id * offset_strides[1] + 1 * offset_strides[2]);

		// outside of image gives large constant value
		float depth_at_x = depth_background_val;

		// depth values at (u, v)
		if ((0 <= u && u < cols) && (0 <= v && v < rows)){
			depth_at_x = *reinterpret_cast<float*>(depthmaps.get_data() + frame_id * depth_strides[0] + v * depth_strides[1] + u * depth_strides[2]);
		}

		short t1_div_x[2] = { static_cast<short>(std::floor(u + (t1_c[0] / depth_at_x) + 0.5)), static_cast<short>(std::floor(v + (t1_c[1] / depth_at_x) + 0.5)) };
		short t2_div_x[2] = { static_cast<short>(std::floor(u + (t2_c[0] / depth_at_x) + 0.5)), static_cast<short>(std::floor(v + (t2_c[1] / depth_at_x) + 0.5)) };

		// outside of image gives large constant value
		float depth_at_x_t1 = depth_background_val;
		float depth_at_x_t2 = depth_background_val;

		// depth values at(offsets + t1 / d(offsets)) and(offsets + t2 / d(offsets))
		if ((0 <= t1_div_x[0] && t1_div_x[0] < cols) && (0 <= t1_div_x[1] && t1_div_x[1] < rows)){
			depth_at_x_t1 = *reinterpret_cast<float*>(depthmaps.get_data() + frame_id * depth_strides[0] + t1_div_x[1] * depth_strides[1] + t1_div_x[0] * depth_strides[2]);
		}
		if ((0 <= t2_div_x[0] && t2_div_x[0] < cols) && (0 <= t2_div_x[1] && t2_div_x[1] < rows)){
			depth_at_x_t2 = *reinterpret_cast<float*>(depthmaps.get_data() + frame_id * depth_strides[0] + t2_div_x[1] * depth_strides[1] + t2_div_x[0] * depth_strides[2]);
		}

		*reinterpret_cast<bool*>(partition.get_data() + i * partition_strides[0]) = (depth_at_x_t1 - depth_at_x_t2) < tau;
	}

	return partition;
}


BOOST_PYTHON_MODULE(pyrdtree_wrapper){
	Py_Initialize();
	np::initialize();
	boost::python::def("_calc_partition_c", _calc_partition_c);
}